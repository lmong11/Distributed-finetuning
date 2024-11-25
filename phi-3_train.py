import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, set_seed
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import tracemalloc
import psutil
import argparse
import numpy as np
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# Set taskid via command line
parser = argparse.ArgumentParser(description="Cluster ID argument settings")
parser.add_argument("--cluster_id", type=int, default=0, help="Select the cluster value to filter the dataset")
cli_args = parser.parse_args()

# Set the global parameters
model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_name = "lmong/clustered_oss_dataset_cleaned"
dataset_split = "train"
new_model = f"single_adapter_task{cli_args.cluster_id}_phi-3"
hf_model_repo = f"dxltt1211/{new_model}"
hf_adapter_repo = f"dxltt1211/task{cli_args.cluster_id}_singleadapter"
device_map = {"": 0}
use_4bit = True
bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_4bit_quant_type = "nf4"
use_double_quant = True
lora_r, lora_alpha, lora_dropout = 16, 16, 0.05
target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
compute_dtype = torch.float16

set_seed(1234)

# Load and filter dataset
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.filter(lambda example: example['cluster'] == cli_args.cluster_id)

# 目前只用 30% 数据（记得取消！！！）
subset_size = int(len(dataset) * 0.3)
dataset = dataset.select(range(subset_size))

print(f"Dataset size: {len(dataset)}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# Dataset format functions
def create_message_column(row):
    return {"messages": [{"content": f"{row['prompt']}\n Instruction: {row['instruction']}", "role": "user"},
                         {"content": row['response'], "role": "assistant"}]}


def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


# Format dataset
formatted_dataset = dataset.map(create_message_column).map(format_dataset_chatml)



dataset_split = formatted_dataset.train_test_split(test_size=0.2, seed=1234)
print(f"Train size: {len(dataset_split['train'])}, Test size: {len(dataset_split['test'])}")

# Load model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    bnb_4bit_use_double_quant=use_double_quant
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=bnb_4bit_compute_dtype,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# Initialize W&B
wandb.init(project="phi3", name=f"phi3_single_adapter_task{cli_args.cluster_id}")

# Train adapter
print("Training Adapter...")
output_dir = f"./phi-3-mini-QLoRA/task{cli_args.cluster_id}_singleadapter/"

start_time = time.time()

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules
)
### see if the train size can be increased to 16
### turn off the evaluation loss
args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    do_eval=False,
    optim="adamw_torch",
    per_device_train_batch_size=16, # smaller than 16
    per_device_eval_batch_size=8, # smaller than 16
    gradient_accumulation_steps=8, #bigger than 4
    logging_steps=100,
    learning_rate=2e-5, # smaller than 1e-4
    num_train_epochs=4, # bigger than 3
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    weight_decay=0.01, # added
    report_to="wandb",
    seed=42
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_split['train'],
    eval_dataset=dataset_split['test'],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=args
)

trainer.train()
trainer.save_model(hf_adapter_repo)

end_time = time.time()
training_duration = end_time - start_time
print(f"Adapter training completed in {training_duration:.2f} seconds.")

###### Evaluation ######

# # Clear Cached Memory
# del model
# del trainer
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# # Create a folder to save results
# output_dir = "rouge_results_new"
# os.makedirs(output_dir, exist_ok=True)

# # Load ROUGE evaluator
# rouge_metric = evaluate.load("rouge")

# def test_inference(pipe, prompt):
#     prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
#     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180)
#     return outputs[0]['generated_text'][len(prompt):].strip()

# def calculate_rouge(pipe, row):
#     response = test_inference(pipe, row['messages'][0]['content'])
#     result = rouge_metric.compute(predictions=[response], references=[row['response']], use_stemmer=True)
#     result = {key: value * 100 for key, value in result.items()}
#     result['response'] = response
#     return result

# # Evaluate base model
# file_path = os.path.join(output_dir, f"rouge_task{cli_args.cluster_id}_single_phi-3.txt")
# if not os.path.exists(file_path):
#     print("Processing original Phi-3 model...")
#     model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=compute_dtype)
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
#     metrics = dataset_split['test'].map(lambda row: calculate_rouge(pipe, row), batched=False)

#     rouge1_mean = np.mean(metrics['rouge1'])
#     rouge2_mean = np.mean(metrics['rouge2'])
#     rougel_mean = np.mean(metrics['rougeL'])
#     rougelsum_mean = np.mean(metrics['rougeLsum'])
    
#     with open(file_path, "w") as f:
#         f.write(f"Base Model - Rouge 1 Mean: {rouge1_mean}\n")
#         f.write(f"Base Model - Rouge 2 Mean: {rouge2_mean}\n")
#         f.write(f"Base Model - Rouge L Mean: {rougel_mean}\n")
#         f.write(f"Base Model - Rouge Lsum Mean: {rougelsum_mean}\n")

#     print(f"Results saved to {file_path}")
#     del model
#     del pipe
#     torch.cuda.empty_cache()
#     gc.collect()

# # Evaluate trained adapter
# print(f"Processing adapter model...")
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=compute_dtype)
# model = PeftModel.from_pretrained(model, hf_adapter_repo)
# model = model.merge_and_unload()

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
# metrics = dataset_split['test'].map(lambda row: calculate_rouge(pipe, row), batched=False)

# rouge1_mean = np.mean(metrics['rouge1'])
# rouge2_mean = np.mean(metrics['rouge2'])
# rougel_mean = np.mean(metrics['rougeL'])
# rougelsum_mean = np.mean(metrics['rougeLsum'])

# output_file = os.path.join(output_dir, f"rouge_phi3_task{cli_args.cluster_id}_singleadapter.txt")
# with open(output_file, "w") as f:
#     f.write(f"Adapter - Rouge 1 Mean: {rouge1_mean}\n")
#     f.write(f"Adapter - Rouge 2 Mean: {rouge2_mean}\n")
#     f.write(f"Adapter - Rouge L Mean: {rougel_mean}\n")
#     f.write(f"Adapter - Rouge Lsum Mean: {rougelsum_mean}\n")

# print(f"Results saved to {output_file}")
