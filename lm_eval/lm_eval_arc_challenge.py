import csv
import torch
import string
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


class Arc_Challenge:
    def __init__(self):
        self.dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="test")

    def eval(self, model, tokenizer):
        model_results = []
        model_results_norm = []
        label_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, '1':0, '2':1, '3':2, '4':3}
        for data in tqdm(self.dataset):
            prompt1 = "Question: " + data['question'] + "\nAnswer:"
            tokenized_input1 = tokenizer(prompt1, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)

            ##### Storing the loglikelihood of each member in the list 
            loglokelihoods = []
            choice_lens = []
            true_output = label_map[data['answerKey']]
            for choice in data['choices']['text']:
                ##### Answer Tokenization
                choice_lens.append(len(choice))
                prompt2 = ' ' + choice
                prompt2 = prompt2.rstrip(string.punctuation)
                tokenized_input2 = tokenizer(prompt2, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
                
                ##### lengths
                GT_len = tokenized_input2.shape[1]
                model_inp_len = tokenized_input1.shape[1] + GT_len - 1

                ##### Model Prompt generation
                tokenized_input = torch.cat((tokenized_input1, tokenized_input2[:, :GT_len-1]), dim=1)

                ##### Model Output
                output = model(tokenized_input).logits
                multi_logits = F.log_softmax(output, dim=-1)

                logits = multi_logits[:, model_inp_len - GT_len: model_inp_len, :]
                logits = torch.gather(logits, 2, tokenized_input2.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                loglokelihoods.append(float(logits.sum()))

            pred_output_choice = np.argmax(loglokelihoods)
            pred_output_choice_norm = np.argmax(list(np.array(loglokelihoods) / np.array(choice_lens)))

            if pred_output_choice == true_output: model_results.append(1.0)
            else: model_results.append(0.0)
            if pred_output_choice_norm == true_output: model_results_norm.append(1.0)
            else: model_results_norm.append(0.0)

        return round(np.sum(model_results) / len(model_results), 4), round(np.sum(model_results_norm) / len(model_results_norm), 4)




