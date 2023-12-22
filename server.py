"""
OpenDI HyperCycle Hackathon 2023
Challenge 3: Genetic Algorithm for Automated Prompt Engineering
"""
import os
import json

from pyhypercycle_aim import SimpleServer, JSONResponseCORS, aim_uri

from ga import GA
from mutator import Mutator
from dataset import Dataset
from template import Template

PORT = os.environ.get("PORT", 4002)


class GeneticWrapper(SimpleServer):
    manifest = {"name": "CBGeneticAlgo",
                "short_name": "cb-gen",
                "version": "0.1",
                "license": "MIT",
                "author": "TeamAC"
                }

    def __init__(self):
        pass

    @aim_uri(uri="/prompt", methods=["POST"],
             endpoint_manifest={
                 "input_query": "",
                 "input_headers": "",
                 "output": {},
                 "documentation": "Returns the prompt and the score based on the desired output",
                 "example_calls": [{
                     "body": {"data": [{"input": "How would a typical person answer each of the following questions about causation?\nA machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit?\nOptions:\n- Yes\n- No", "target": "No"}]},
                     "method": "POST",
                     "query": "",
                     "headers": "",
                     "output": {"prompt": "string", "score": 0.004}
                 }]
             })
    async def prompt(self, request):
        # define the total iterations
        n_iter = 100
        # define the population size
        n_pop = 100
        # define sample size to used during training
        sample_size = 4

        prompt_template = "[context]\n\nQuestion: [question]\nAnswer: Let's think step by step."
        demo_template = "Question: [question]\nAnswer: [answer]"
        template = Template(prompt_template, demo_template)
        request_json = await request.json()
        dt = request_json['data']
        dataset = Dataset(format = "dict", dict_data = json.loads(dt), template = template)
        mutator = Mutator()
        ga = GA("t5-small", dataset = dataset, mutator = mutator, num_iter = n_iter, k = n_pop)

        # perform the genetic algorithm search
        best_prompt, score = ga.save_best_prompt()
        return JSONResponseCORS({"prompt": best_prompt, "score": score})


def main():
    # example usage:
    app = GeneticWrapper()
    app.run(uvicorn_kwargs={"port": PORT, "host": "0.0.0.0"})


if __name__ == '__main__':
    main()