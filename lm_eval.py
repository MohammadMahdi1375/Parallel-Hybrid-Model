import os
import sys
import dill
sys.path.append(os.path.abspath("./lm_eval"))
import torch
from lm_eval_arc_easy import Arc_Easy
from lm_eval_arc_challenge import Arc_Challenge
from lm_eval_boolq import BoolQ
from lm_eval_hellaswag import Hellaswag
from lm_eval_lambada import Lambada
from lm_eval_piqa import PIQA
from lm_eval_winogrande import Winogrande
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
DEVICE="cuda"

repo_name = "ai21labs/AI21-Jamba-1.5-Mini"
tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = torch.load("/home/m_m58330/Hybrid_Model/transformers/src/transformers/models/parallel_jamba/saved_model/SHM_1B/model.pth", weights_only=False)

model = model.to(DEVICE)


data_list = [Arc_Easy(), Arc_Challenge(), BoolQ(), Hellaswag(), Lambada(), PIQA(), Winogrande()]

for lm_eval in data_list:
    results = lm_eval.eval(model, tokenizer)
    data_name = lm_eval.__class__.__name__

    if data_name in ["Arc_Easy", "Arc_Challenge", "Hellaswag", "PIQA"]:
        print(data_name, "|"*(35-len(data_name)))
        print(f"|||||| Accuracy      = {results[0]} ||||||")        
        print(f"|||||| Accuracy_Norm = {results[1]} ||||||")
        print("|"*36)

    elif data_name=="Lambada":
        print(data_name, "|"*(31-len(data_name)))
        print(f"|||||| Standard Accuracy   = {results[0]} ||||||")        
        print(f"|||||| Openai Accuracy     = {results[2]} ||||||")
        print(f"|||||| Total Accuracy      = {results[4]} ||||||")        
        print(f"|||||| Standard Perplexity = {results[1]} ||||||")
        print(f"|||||| Openai Perplexity   = {results[3]} ||||||")
        print(f"|||||| Total Perplexity    = {results[5]} ||||||")
        print("|"*42)
        
    elif data_name in ["BoolQ", "Winogrande"]:
        print(data_name, "|"*(30-len(data_name)))
        print(f"|||||| Accuracy = {results} ||||||") 
        print("|"*31)


