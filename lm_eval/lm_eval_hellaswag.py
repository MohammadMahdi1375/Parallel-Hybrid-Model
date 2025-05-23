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
query = ""
for idx, data in enumerate(cloned_reqs):
    data_list = []
    if (data.doc['query'] != query):
        ind = data.doc['ind']
        query = data.doc['query']
        choices = data.doc['choices']
        label = data.doc['label']
        data_list.append(ind)
        data_list.append(query)
        data_list.append(choices)
        data_list.append(label)
        ds.append(data_list)
df = pd.DataFrame(ds, columns=columns)
df.to_json('/home/mohammad-m/TTT/RL/lm_eval_data/hellaswag_validation.json', orient="records", lines=True)
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


class Hellaswag:
    def __init__(self):
        self.dataset = pd.read_json("./lm_eval/lm_eval_data/hellaswag_validation.json", orient='records', lines=True)

    def eval(self, model, tokenizer):
        model_results = []
        model_results_norm = []
        for idx in tqdm(range(len(self.dataset)), desc="Processing"):
            loglokelihoods = []
            choice_lens = []
            true_output = self.dataset['label'][idx]
            if True:
                for choice in self.dataset['Choices'][idx]:
                    choice_lens.append(len(choice))
                    prompt = self.dataset['Query'][idx] + ' ' + choice
                    tokenizer_choice = tokenizer(' ' + choice, return_tensors='pt', add_special_tokens=False).to(model.device)
                    tokenized = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
                    
                    context_length = tokenized['input_ids'].shape[1] - 1
                    choice_length = tokenizer_choice['input_ids'].shape[1]

                    output = model(tokenized['input_ids']).logits
                    multi_logits = F.log_softmax(output, dim=-1)

                    logits = multi_logits[:, context_length - choice_length: context_length, :]
                    logits = torch.gather(logits, 2, tokenizer_choice['input_ids'].unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    loglokelihoods.append(float(logits.sum()))
                
                pred_output_choice = np.argmax(loglokelihoods)
                pred_output_choice_norm = np.argmax(list(np.array(loglokelihoods) / np.array(choice_lens)))

                if pred_output_choice == true_output: model_results.append(1.0)
                else: model_results.append(0.0)
                if pred_output_choice_norm == true_output: model_results_norm.append(1.0)
                else: model_results_norm.append(0.0)

        return round(np.sum(model_results) / len(model_results), 4), round(np.sum(model_results_norm) / len(model_results_norm), 4)