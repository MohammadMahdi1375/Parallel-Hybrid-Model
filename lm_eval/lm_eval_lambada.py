""""
In orderr to process the 'Hellaswag' data for passing to the model, add the following code to this address:
./lm-evaluation-harness/lm_eval/evaluator.py ----> <<evaluate>> function ----> before 
                                                   resps = getattr(lm, reqtype)(cloned_reqs)

    1) Query: Main Context
    2) choices: options
    3) label: true answer
    4) tokenizer(Query) + tokenizer(choice)[:-2]: The input of the model
    5) Sum up the the likelihoods frm last 
columns = ['ind', 'Query', 'Choices', 'label']
ds = []
for idx, data in enumerate(requests['loglikelihood']):
            data_list = []
            data_list.append(idx)
            data_list.append(data.args[0])
            data_list.append(data.args[1])
            ds.append(data_list)

        df = pd.DataFrame(ds, columns=columns)
        df.to_json('/home/mohammad-m/TTT/RL/lm_eval_data/lambada_validation.json', orient="records", lines=True)
"""


import re
import csv
import torch
import string
import numpy as np
import pandas as pd
from collections import Counter
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline


class Lambada:
    def __init__(self):
        self.dataset = pd.read_json("./lm_eval/lm_eval_data/lambada_validation.json", orient='records', lines=True)

    def eval(self, model, tokenizer):
        openai_lambada_accuracy = []
        openai_lambada_perplexity = []
        standard_lambada_accuracy = []
        standard_lambada_perplexity = []

        for idx in tqdm(range(len(self.dataset)), desc="Processing"):
            loglokelihoods = []
            true_output = self.dataset['label'][idx]
            
            prompt = self.dataset['Query'][idx] + true_output
            tokenizer_choice = tokenizer(true_output, return_tensors='pt', add_special_tokens=False).to(model.device)
            tokenized = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            
            context_length = tokenized['input_ids'].shape[1] - 1
            choice_length = tokenizer_choice['input_ids'].shape[1]

            output = model(tokenized['input_ids']).logits
            multi_logits = F.log_softmax(output, dim=-1)

            logits = multi_logits[:, context_length - choice_length: context_length, :]
            greedy_tokens = logits.argmax(dim=-1)
            logits = torch.gather(logits, 2, tokenizer_choice['input_ids'].unsqueeze(-1)).squeeze(-1)  # [1, seq]

            ## <<Standard>> Lambada
            if self.dataset['ind'][idx] < 5153:
                if (greedy_tokens == tokenizer_choice['input_ids']).all(): standard_lambada_accuracy.append(1.0)
                else: standard_lambada_accuracy.append(0.0)
                standard_lambada_perplexity.append(float(logits.sum()))
            ## <<openai>> Lambada
            else:
                if (greedy_tokens == tokenizer_choice['input_ids']).all(): openai_lambada_accuracy.append(1.0)
                else: openai_lambada_accuracy.append(0.0)
                openai_lambada_perplexity.append(float(logits.sum()))


        standard_acc = np.sum(standard_lambada_accuracy) / len(standard_lambada_accuracy)
        standard_perplex = np.exp(-np.mean(standard_lambada_perplexity))
        openai_acc = np.sum(openai_lambada_accuracy) / len(openai_lambada_accuracy)
        openai_perplex = np.exp(-np.mean(openai_lambada_perplexity))
        total_acc = (standard_acc + openai_acc)/2
        total_perplex = (standard_perplex + openai_perplex)/2
        
        return round(standard_acc, 4), round(standard_perplex, 4), round(openai_acc, 4), round(openai_perplex, 4), round(total_acc, 4), round(total_perplex, 4)
    