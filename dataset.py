import numpy as np
import pandas as pd
import json
# from datasets import load_dataset
from transformers import AutoTokenizer
from template import Template
class Dataset:
    def __init__(self, format = "json", filepath = None, dict_data = None, template: Template = None):
        try:
            if format == "json":
                f = open(filepath)
                data_dict = json.load(f)
                self.df = pd.DataFrame(data_dict["examples"])
            elif format == "csv":
                self.df = pd.read_csv(filepath)
            elif format == "dict":
                if not isinstance(dict_data, list) and isinstance(dict_data["input"], str):
                    d = {k: [v] for k, v in dict_data.items()}
                else: 
                    d = dict_data
                self.df = pd.DataFrame(d)
        except:
            raise FileNotFoundError("File not found or format not supported. Only json and csv is supported.", filepath)
        print(self.df.shape)
        if (template is None) or (not isinstance(template, Template)): 
            raise ValueError("Invalid template. It should be an instance of class Template")
        self.template = template

    def get_data_as_df(self):
        return self.df

    def apply_template_ques(self, prompt, question):
        return self.template.get_llm_input(question = question, context = prompt)

    def apply_template_df(self, prompt, df = None) -> pd.DataFrame:
        if df is None:
            df = self.df.copy()
        df["input"] = df["input"].map(lambda x: self.template.get_llm_input(question = x, context = prompt))
        return df

    def get_random_sample(self, sample_size = 2):
        if sample_size > self.df.shape[0]:
            print("warning sample size largee than total samples. Returning the entire dataset")
            return self.df
        return self.df.sample(sample_size, replace = False).reset_index(drop = True)

    def get_tokenized_df(self, prompt, tokenizer, max_length):
        df = self.apply_template_df(prompt)
        df["input"] = df["input"].map(lambda x: tokenizer(x, padding = "longest"))
        return df

def main():
    file = "./datasets/causal_judgement.json"
    prompt = "Some PRompt. \n aofb \n noafo.. $ aobfoa\n"
    template = Template("./templates/prompt/context", "./templates/demo/context")
    dataset = Dataset(format = "json", filepath = file, template = template)
    print(dataset.get_data_as_df().head())
    print(dataset.apply_template_df("Some PRompt. \n aofb \n noafo.. $ aobfoa\n")["input"][0])
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    df = dataset.get_tokenized_df(prompt, tokenizer, 1024)
    df["input"][0]
    print(df.head())
    dataset.get_random_sample(2)


if __name__ == "__main__":
    main()