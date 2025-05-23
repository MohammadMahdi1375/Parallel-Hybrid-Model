import csv
import time
import torch
import copy
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import torch.nn.functional as F
from configuration_jamba import JambaConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from modeling_jamba import JambaModel, JambaForCausalLM
from calflops import calculate_flops_hf, calculate_flops
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler



##########################################################################################################################################################################
############################################################################## Parameters ################################################################################
##########################################################################################################################################################################
parser = argparse.ArgumentParser(description="Define hyperparameters for the model.")

parser.add_argument("--d_model",                        default=768,            help="Model dimentionality (d_model=hidden_size)")
parser.add_argument("--MLP_intermediate_size",          default=2048,           help="diemntionality of the inn and out in the gate, up, and down projections")
parser.add_argument("--num_attention_heads",            default=32,             help="Number of Attention Heads")
parser.add_argument("--num_key_value_heads",            default=8,              help="Number of Key Value Heads")
parser.add_argument("--N_Hybrid_Layers",                default=2,              help="Number of Blocks in the model")
parser.add_argument("--Attention_OFFSET",               default=1,              help="Layer Index from which Attetion starts in the model")
parser.add_argument("--Attetnion_PERIOD",               default=2,              help="Period of Attention repitition")
parser.add_argument("--MAX_LENGTH",                     default=1024,           help="Maximum number of tokens are fed to the model for training")
parser.add_argument("--TOKEN_LIMIT",                    default=100_000,  help="Number of tokens we want to train our model on")
parser.add_argument("--BATCH_SIZE",                     default=2,             help="Batch size")
parser.add_argument("--gradient_accumulation_steps",    default=1,              help="Gradient Accumulation Step")
parser.add_argument("--LR",                             default=3e-4,           help="Learning Rate")
parser.add_argument("--WEIGHT_DECAY",                   default=0.1,            help="weight decay in the optimizer")
parser.add_argument("--betas",                          default=(0.9, 0.95),    help="weight decay in the optimizer")
parser.add_argument("--DEVICE",                         default='cuda',        help="Device Type")
parser.add_argument("--model_adr",                      default='PHM_135_test',     help="Folder in which we store the final model")
parser.add_argument("--Parallel_Mode",                  default='D',            help="Folder in which we store the final model")
args = parser.parse_args()

total_token_count = 0
tensorboard_writer = SummaryWriter(log_dir="runs_1B/"+args.model_adr + "_" + args.Parallel_Mode)
##########################################################################################################################################################################
############################################################################# Model Declration ###########################################################################
##########################################################################################################################################################################
config = JambaConfig(
    hidden_size=args.d_model,
    intermediate_size=args.MLP_intermediate_size,
    num_attention_headsv=args.num_attention_heads,
    num_key_value_heads=args.num_key_value_heads,
    Prallel_Mode=args.Parallel_Mode,
    num_hidden_layers=args.N_Hybrid_Layers,
    num_experts_per_tok=0,
    num_experts=0,
    expert_layer_period=1,
    expert_layer_offset=0,
    attn_layer_period=args.Attetnion_PERIOD,
    attn_layer_offset=args.Attention_OFFSET,
)
model = JambaForCausalLM(config)
model = model.to(args.DEVICE)

tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-1.5-Mini")
tokenizer.save_pretrained("./saved_model/" + args.model_adr + "/")
print("=============================================================================")
print(model)
print(f"The total Number of Parameters in the model: {sum(p.numel() for p in model.parameters())}")
##########################################################################################################################################################################
############################################################################ Hyperparameters #############################################################################
##########################################################################################################################################################################
dataset = load_dataset("DKYoon/SlimPajama-6B", split="train")
dataloader = DataLoader(dataset, batch_size=args.BATCH_SIZE)

num_training_steps = 0
for idx, data in tqdm(enumerate(dataloader)):
    inputs = tokenizer(data['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=args.MAX_LENGTH).to(args.DEVICE)
    total_token_count += torch.sum(inputs['attention_mask'] == 1).item()
    if total_token_count >= args.TOKEN_LIMIT: 
        break
    num_training_steps += 1

print("==========================================================================================")
print(num_training_steps)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, betas=args.betas, weight_decay=args.WEIGHT_DECAY)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps/args.gradient_accumulation_steps),  # 10% warmup
    num_training_steps=int(num_training_steps/args.gradient_accumulation_steps)  # No need to predefine total steps
)


csv_losses = "./" + args.model_adr + "_" + args.Parallel_Mode + "_loss.csv"
with open(csv_losses, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['iter', 'loss'])


##########################################################################################################################################################################
################################################################################ Trainer #################################################################################
##########################################################################################################################################################################
optimizer.zero_grad()
model.train()
total_token_count = 0
total_time = 0

gradient_period_loss = []

gpu_total_s, gpu_steps = 0.0, 0

data = next(iter(dataloader))
inputs = tokenizer(data['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=args.MAX_LENGTH).to(args.DEVICE)
flops, macs, params = calculate_flops(model=model, 
                                              kwargs = inputs,
                                              include_backPropagation=True,
                                              print_results=False
                                            )
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
                
progress_bar = tqdm(enumerate(dataloader), desc='Processing')
for idx, data in progress_bar:
    start_time = time.time()
    inputs = tokenizer(data['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=args.MAX_LENGTH).to(args.DEVICE)
    total_token_count += torch.sum(inputs['attention_mask'] == 1).item()

    if total_token_count < args.TOKEN_LIMIT:
        start_event = time.time()

        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        gradient_period_loss.append(loss.item())
        loss.backward()

        if (idx % args.gradient_accumulation_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            end_event = time.time()
            if (idx > int(0.5 * num_training_steps/args.gradient_accumulation_steps)):
                gpu_total_s += (end_event - start_event)
                gpu_steps += 1

            total_time += (time.time() - start_time)

            tensorboard_writer.add_scalar("Training Loss", sum(gradient_period_loss), total_token_count)
            tensorboard_writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], total_token_count)

            with open(csv_losses, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_token_count] + [round(sum(gradient_period_loss), 3)])

            gradient_period_loss = []

        else:
            total_time += (time.time() - start_time)

        if idx % int(0.05 * num_training_steps) == 0:
            torch.save(model, "./saved_model/" + args.model_adr + "_" + args.Parallel_Mode + "/model.pth")
            print(f"Model was saved in token number: {total_token_count}")

        progress_bar.set_postfix(loss=sum(gradient_period_loss), Token_num=total_token_count, lr=optimizer.param_groups[0]['lr'])
        
    else:
        break

tensorboard_writer.close()
print("==========================================================================================")
print(f"Token/Sec is: {round(total_token_count / total_time, 3)}")
print(f"Iter/Sec is: {round(num_training_steps / total_time, 3)}")
avg_gpu_time = (gpu_total_s / gpu_steps)
print(f"Avg GPU time per step: {avg_gpu_time:.4f}s")
flops_splits = flops.split(" ")
print(flops_splits)

if (flops_splits[1] == "GFLOPS"):
    print(f"MFU is: {float(flops_splits[0]) / (avg_gpu_time * 45.158 * 1000)}")
elif (flops_splits[1] == "TFLOPS"):
    print(f"MFU is: {float(flops_splits[0]) / (avg_gpu_time * 45.158)}")
# torch.save(model, "./saved_model/" + args.model_adr + "_" + args.Parallel_Mode + "/model.pth")

# print("Model Saved")