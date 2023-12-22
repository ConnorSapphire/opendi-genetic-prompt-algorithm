# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from template import Template
from mutator import Mutator
from dataset import Dataset
from utils import extract_answer, encode_answer, get_model, get_tokenizer
import numpy as np
import pandas as pd
import torch

from pprint import pprint

class GA:
    supported_models = ["t5-small", "t5-base", "t5-large", "gpt2"]
    def __init__(self, model, dataset: Dataset, mutator, fitness_func = None, num_iter = 100, k = 5, select_method = "random", max_length = 512, batch_size = 2, sample_size = None):
        if model not in self.supported_models:
            raise ValueError("Model mot suported. Supported models are", self.supported_models)
        elif model == "t5-small":
            checkpoint = "google/flan-t5-small"
        elif model == "t5-base":
            checkpoint = "google/flan-t5-base"
        elif model == "t5-large":
            checkpoint = "google/flan-t5-large"
        else:
            checkpoint = "distilgpt2"

        self.max_length = max_length
        self.model = get_model(checkpoint)
        self.tokenizer = get_tokenizer(checkpoint)

        self.fitness_function = fitness_func


        # self.template = Template(template, demo_template)
        self.k = k
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.select_method = "random"
        self.has_run = False
        self.mutator = mutator
        self.dataset = dataset
        if sample_size is None:
            self.df = dataset.get_data_as_df()
        else:
            self.df = dataset.get_random_sample(sample_size)
        self.init_prompt = open("./initial_prompts/causal_judgement").read()
        self.init_prompt = {
                "prompt": self.init_prompt,
                "score": 0
            }
        # self.pop = [
        #     self.prompt
        # ]

    def select_random(self, pop):
        return np.random.randint(0, len(pop))

    def select_weighted(self, pop):
        if len(pop) == 1:
            return 0
        else:
            return np.random.randint(0, len(pop), p = self.scores)

    def select_prompt(self, pop):
        if self.select_method == "random":
            return self.select_random(pop)
        elif self.select_method == "weighted":
            return self.select_weighted(pop)


    def get_best_candidate(self):
        return sorted(self.pop, key = lambda x: x["score"], reverse = True)[0]

    def get_sentence_distance(self, a, b):
        ea = self.mutator.encode_sentences(a)
        eb = self.mutator.encode_sentences(b)
        return self.mutator.get_sentence_distance(ea, eb)

    def select_crossover_point(self, a, b):
        max_sim = 10
        i = 0
        n = min(len(a), len(b))
        pos = np.random.randint(0, n - 1)
        while self.get_sentence_distance(a[pos], b[pos]) < 0.5 and i < max_sim:
            pos = np.random.randint(0, n - 1)
            i += 1
        return pos

    def convert_to_prompt(self, sentences):
        return " ".join(sentences)

    def get_children(self, a, b):
        sa = self.mutator.get_sentences(a)
        sb = self.mutator.get_sentences(b)
        pos = self.select_crossover_point(sa, sb)
        return self.convert_to_prompt(sa[:pos] + sb[pos:]), self.convert_to_prompt(sb[:pos] + sa[pos:])

    def get_prompt_dict(self, prompt):
        return {
            "prompt": prompt,
            "score": 0
        }

    def crossover(self):
        best_prompt = self.get_best_candidate()

        new_pop = [best_prompt]
        for i in range(self.k // 2):
            a, b = np.random.choice(self.pop, size = 2, replace = False)
            ca, cb = self.get_children(a["prompt"], b["prompt"])
            ca = self.get_prompt_dict(ca)
            cb = self.get_prompt_dict(cb)
            new_pop.extend([ca, cb])
        return new_pop

    def mutate_prompt(self, prompt):
        mutated_prompt = self.mutator.get_mutated_prompt(prompt["prompt"])
        return self.get_prompt_dict(mutated_prompt)

    def init_population(self):
        self.pop = [self.mutate_prompt(self.init_prompt) for _ in range(self.k)]

    def generate_answers(self, prompt: str, df_orig):
        if df_orig is None:
            df = self.dataset.apply_template_df(prompt, self.df)
        else:
            df = self.dataset.apply_template_df(prompt, df_orig)
        n = df.shape[0]
        inputs = []
        for i in range(n):
            inputs.append(self.tokenizer(df["input"][i], max_length = self.max_length))

        generated_text = []
        for ip in inputs:
            out = self.model.generate(torch.tensor([ip["input_ids"]]), max_new_tokens = 1024)
            r = self.tokenizer.decode(out[0], skip_special_tokens = True)
            generated_text.append(r)
        answers = [extract_answer(s) for s in generated_text]
        answers = np.array(list(map(encode_answer, answers)))
        target = np.array(list(map(encode_answer, df["target"])))
        for i, a in enumerate(answers):
          if a is None:
            answers[i] = !target[i]
        return pd.DataFrame({
            "answers": answers,
            "target": target
        })

    def evaluate_fitness(self, prompt: str, df = None):
        if callable(self.fitness_function):
            try:
                ans = self.fitness_function(prompt, df)
                ans = float(ans)
                return ans
            except AssertionError:
                raise ValueError("The output of the fitness function must be convertible to a numeric value example int or float")
            except AttributeError:
                raise AttributeError("Did you accidently included the parentheses")

        ans_df = self.generate_answers(prompt, df)
        return (ans_df["answers"] == ans_df["target"]).mean()

    def evaluate_population(self, pop):
        for prompt in pop:
            prompt["score"] = self.evaluate_fitness(prompt["prompt"])
        return

    def update_population(self, new_pop):
        # print(type(new_pop))
        self.pop.extend(new_pop)
        self.pop = sorted(self.pop, key = lambda x: x["score"], reverse = True)
        self.pop = self.pop[:self.k]
        # print(self.pop[0])
        # print(type(self.pop[0]))

    def tune_prompt(self):
        self.init_score = self.evaluate_fitness(self.init_prompt["prompt"])
        self.init_population()
        self.evaluate_population(self.pop)
        for i in range(self.num_iter):
            print("iteration:", i)
            new_pop = self.crossover()
            print("crossove done.")
            index = self.select_prompt(new_pop)
            print("selecting and mutating a random prompt")
            init_score = self.evaluate_fitness(new_pop[index]["prompt"]) # expensive
            new_pop[index] = self.mutate_prompt(new_pop[index])
            self.evaluate_population(new_pop) # expensive
            score = new_pop[index]["score"]
            self.mutator.update_mutator(score - init_score)
            self.update_population(new_pop)
            print("Scores:", [p["score"] for p in self.pop])

    def best_prompt(self):
        self.tune_prompt()
        try:
            self.mutator.save_history("history.csv")
        except:
            print("Could not save history")
        max_score = self.pop[0]["score"]
        best_prompt = self.pop[0]
        for p in self.pop:
            if max_score < p["score"]:
                max_score = p["score"]
                best_prompt = p
        return best_prompt

    def save_best_prompt(self, filename = "best_prompt.txt"):
        best_prompt = self.best_prompt()
        print(best_prompt["prompt"])
        initial_score = self.evaluate_fitness(self.init_prompt["prompt"], self.dataset.get_random_sample(50))
        print("Score: ", initial_score)
        best_score = self.evaluate_fitness(best_prompt["prompt"], self.dataset.get_random_sample(50))
        print("Score: ", best_score)
        self.has_run = True
        try:
            f = open(filename, mode = 'w')
            f.write(best_prompt["prompt"])
            f.close()
        except:
            pass
        return best_prompt["prompt"], best_score

    def answer_question(self, question):
        if not self.has_run:
            print("The algorithm has not been run. Using the initial prompt")
            prompt = self.init_prompt["prompt"]
        else:
          prompt = open("best_prompt.txt").read()
        prompt = self.dataset.apply_template_ques(question = question, context = prompt)
        return self

def main():
    template = Template("./templates/prompt/context", "./templates/demo/context")
    dataset = Dataset(format = "json", filepath = "./datasets/causal_judgement.json", template = template)
    mutator = Mutator("chatgpt")
    ga = GA("t5-base", dataset, mutator = mutator, max_length = 1024 + 512, num_iter = 4, sample_size = 4)
    ga.save_best_prompt("best_prompt.txt")

if __name__ == "__main__":
    main()