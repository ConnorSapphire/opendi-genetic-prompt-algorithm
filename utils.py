import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, T5EncoderModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

def clean_prompt(prompt: str):
    """Cleans the prompt before passing it to the model. Removes extra spaces and '.'
    """
    new_prompt = prompt
    new_prompt = re.sub(r"([A-Z])\.", r"\1", new_prompt)
    new_prompt = re.sub(r"[\n]{1,}", "\n", new_prompt)
    return new_prompt

def extract_answer(s: str):
    """Extracts the answer from the string.
    """
    
    answer_match = [i for i in re.finditer(r"answer is (yes|no)\W", s, re.IGNORECASE)]
    if len(answer_match) > 0:
        answer_match = answer_match[-1]
        answer_string = s[answer_match.start(): answer_match.end()]
        # print(answer_string)
        matches = list(re.finditer(r"Yes|No", answer_string, re.IGNORECASE))[-1]
        return answer_string[matches.start(): matches.end()]
    else:
        return None

def encode_answer(a):
    """Encodes the given string. Assumes string is "yes" or "no".
    """
    
    if a is None:
      return None
    a = a.lower()
    if a == "yes":
        return True
    else:
        return False

def get_model(checkpoint, encoder = False):
    """Retrieves the Large Language Model for use.
    """
    if encoder:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # T5EncoderModel.from_pretrained(checkpoint)
    else:
        return AutoModelForSeq2SeqLM.from_pretrained(checkpoint, max_length = 1024 + 512)
        # return AutoModelForCausalLM.from_pretrained(checkpoint, max_length = 1024 + 512)

def get_tokenizer(checkpoint):
    """Retrieves the tokeniser.
    """
    return AutoTokenizer.from_pretrained(checkpoint)

def main():
    s = "afn noaifn niaona  oiandao aoinsdoa oiaod ''a ionad  answer is yes iab oianosid np np no aofbak. so the answer is Yes."
    print(extract_answer(s))
    s = "afn noaifn niaona  oiandao aoinsdoa oiaod ''a ionad  answer is yes iab oianosid np np no aofbak. so the answer is Yesa."
    print(extract_answer(s))
    s = "afn noaifn niaona  \n\n\n\n\noiandao aoinsdoa\n oiaod\n ''a ionad  answer is yesb iab oianosid np np no aofbak. so the answer is Yesa."
    print(extract_answer(s))
    print(s, clean_prompt(s))
    

if __name__ == "__main__":
    main()
