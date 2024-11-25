#pip install -qqq transformers peft trl datasets wandb evaluate pandas matplotlib huggingface_hub bitsandbytes python-dotenv accelerate flash_attn absl-py nltk rouge_score argparse
#huggingface-cli login --token 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
#huggingface-cli whoami
#wandb login  7e43f53f1b364a524ef9d3e012c28bee4b7d261b
#python phi-3_10adapters_taskid.py --cluster_id 1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, set_seed
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # the program can only use the first GPU; must set before importing any other libraries
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
cli_args  = parser.parse_args()

# Set the global parameters
model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_name = "lmong/clustered_oss_dataset"
dataset_split = "train"
num_adapters = 10
new_model = f"avg10adapters_task{cli_args.cluster_id}_phi-3"  
hf_model_repo = f"dxltt1211/{new_model}" # local repository
hf_adapter_repo = f"dxltt1211/task{cli_args.cluster_id}_adapter" # local repository
device_map = {"": 0}
use_4bit = True
bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_4bit_quant_type = "nf4"
use_double_quant = True
lora_r, lora_alpha, lora_dropout = 16, 16, 0.05
target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
compute_dtype = torch.float16

set_seed(1234)

# Load, filter and seperate dataset
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.filter(lambda example: example['cluster'] == cli_args.cluster_id)  # cluster_id

# 目前只用 30% 数据（记得取消！！！）
subset_size = int(len(dataset) * 0.3)
dataset = dataset.select(range(subset_size))

chunk_size = len(dataset) // num_adapters
adapter_datasets = [dataset.select(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_adapters)]
print(f"Subdataset sizes: {[len(ds) for ds in adapter_datasets]}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# Dataset format functions
def create_message_column(row):
    return {"messages": [{"content": f"{row['instruction']}\n Input: {row['prompt']}", "role": "user"},
                         {"content": row['response'], "role": "assistant"}]}

def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

chatml_datasets = [dataset.map(create_message_column).map(format_dataset_chatml) for dataset in adapter_datasets]
chatml_datasets_split = []
for i, dataset in enumerate(chatml_datasets):
    split = dataset.train_test_split(test_size=0.2, seed=1234)
    chatml_datasets_split.append(split)
    print(f"Adapter {i + 1} - Train size: {len(split['train'])}, Test size: {len(split['test'])}")

# Load model with quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=use_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type,
                                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype, bnb_4bit_use_double_quant=use_double_quant)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=bnb_4bit_compute_dtype, quantization_config=bnb_config, device_map=device_map, trust_remote_code=True)
model = prepare_model_for_kbit_training(model)

# Initialize W&B
wandb.init(project="phi3", name="phi3")

# Train each adapter
adapters = []
training_times = []
for i in range(num_adapters):
    print(f"Training Adapter {i + 1}...")
    output_dir = f"./phi-3-mini-QLoRA/task{cli_args.cluster_id}_adapter{i}/"
    adapter_path = f"{hf_adapter_repo}{i}"

    # Check if the adapter already exists
    if os.path.exists(output_dir):
        print(f"Adapter {i + 1} already exists. Skipping training.")
        # Load the pre-trained adapter and append it to the adapters list
        model = PeftModel.from_pretrained(model, adapter_path)
        adapters.append(model)
    else:
        start_time = time.time()
        
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                                 task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
    
        args = TrainingArguments(output_dir=output_dir, evaluation_strategy="steps", do_eval=True,
                                 optim="adamw_torch", per_device_train_batch_size=16, per_device_eval_batch_size=16,
                                 gradient_accumulation_steps=4, logging_steps=100, learning_rate=1e-4, num_train_epochs=3,
                                 warmup_ratio=0.1, lr_scheduler_type="linear", report_to="wandb", seed=42)
    
        trainer = SFTTrainer(model=model, train_dataset=chatml_datasets_split[i]['train'], eval_dataset=chatml_datasets_split[i]['test'], peft_config=peft_config, dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer, args=args)
    
        trainer.train()
        adapter_path = f"{hf_adapter_repo}{i}" 
        trainer.save_model(adapter_path)
        adapters.append(trainer.model)
    
        end_time = time.time()
        training_duration = end_time - start_time
        training_times.append(training_duration)
        print(f"Adapter {i + 1} training completed in {training_duration:.2f} seconds.")

# Merge Adapters
def merge_adapters(adapters):
    merged_adapter = adapters[0]
    for adapter in adapters[1:]:
        for (name1, param1), (name2, param2) in zip(merged_adapter.named_parameters(), adapter.named_parameters()):
            if name1 == name2:
                param1.data.copy_((param1.data + param2.data) / 2.0)
    return merged_adapter

merged_adapter = merge_adapters(adapters)
merged_adapter.save_pretrained(f"dxltt1211/task{cli_args.cluster_id}_avgadapters")
# model.push_to_hub(f"dxltt1211/avg10adapters_task{cli_args.cluster_id}_phi-3/")
# tokenizer.push_to_hub(f"dxltt1211/avg10adapters_task{cli_args.cluster_id}_phi-3/")




###### Evaluation ######

# Clear Cached Memory
del model
# del trainer
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=True) 

# Create a folder to save results
output_dir = "rouge_results_new"
os.makedirs(output_dir, exist_ok=True)  

# Test dataset
chatml_datasets = dataset.map(create_message_column).map(format_dataset_chatml).train_test_split(test_size=0.2)

# Load ROUGE evaluator
rouge_metric = evaluate.load("rouge")
def test_inference(pipe, prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180)
    return outputs[0]['generated_text'][len(prompt):].strip()
def calculate_rouge(pipe, row):
    response = test_inference(pipe, row['messages'][0]['content'])
    result = rouge_metric.compute(predictions=[response], references=[row['response']], use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    result['response'] = response
    return result




# For phi-3:
file_path = os.path.join(output_dir, f"rouge_task{cli_args.cluster_id}_phi-3.txt")
if not os.path.exists(file_path):
    print("Processing original Phi-3 model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=compute_dtype)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    metrics = chatml_datasets['test'].map(lambda row: calculate_rouge(pipe, row), batched=False)

    rouge1_mean = np.mean(metrics['rouge1'])
    rouge2_mean = np.mean(metrics['rouge2'])
    rougel_mean = np.mean(metrics['rougeL'])
    rougelsum_mean = np.mean(metrics['rougeLsum'])
    output_file = os.path.join(output_dir, f"rouge_task{cli_args.cluster_id}_phi-3.txt")
    with open(output_file, "w") as f:
        f.write(f"Base Model - Rouge 1 Mean: {rouge1_mean}\n")
        f.write(f"Base Model - Rouge 2 Mean: {rouge2_mean}\n")
        f.write(f"Base Model - Rouge L Mean: {rougel_mean}\n")
        f.write(f"Base Model - Rouge Lsum Mean: {rougelsum_mean}\n")

    print(f"Results saved to {output_file}")
    del model
else:
    print("ROUGE results for original Phi-3 model already exist. Skipping.")



# Iterate over all adapters for merging and evaluation
adapter_path_template = f"./dxltt1211/task{cli_args.cluster_id}_adapter{{}}"
for i in range(10):    
    file_path = os.path.join(output_dir, f"rouge_phi3_task{cli_args.cluster_id}_adapter{i}.txt")
    if os.path.exists(file_path):
        print(f"Adapter {i} is already evaluated. Skipping.")
    else:
        print(f"Processing adapter {i}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=compute_dtype)
        
        # Merge the corresponding adapter
        adapter_path = adapter_path_template.format(i)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        
        # Create the text generation pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        pipe.tokenizer.apply_chat_template([{"role": "user", "content": chatml_datasets['test'][0]['messages'][0]['content']}], tokenize=False, add_generation_prompt=True)
        # Compute ROUGE scores
        metrics = chatml_datasets['test'].map(lambda row: calculate_rouge(pipe, row), batched=False)
    
        # Print and save the results
        rouge1_mean = np.mean(metrics['rouge1'])
        rouge2_mean = np.mean(metrics['rouge2'])
        rougel_mean = np.mean(metrics['rougeL'])
        rougelsum_mean = np.mean(metrics['rougeLsum'])
    
        # Save the results to a file
        output_file = os.path.join(output_dir, f"rouge_phi3_task{cli_args.cluster_id}_adapter{i}.txt")
        with open(output_file, "w") as f:
            f.write(f"Adapter {i} - Rouge 1 Mean: {rouge1_mean}\n")
            f.write(f"Adapter {i} - Rouge 2 Mean: {rouge2_mean}\n")
            f.write(f"Adapter {i} - Rouge L Mean: {rougel_mean}\n")
            f.write(f"Adapter {i} - Rouge Lsum Mean: {rougelsum_mean}\n")
    
        print(f"Results saved to {output_file}")
        if 'model' in locals():
            del model
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()
        gc.collect()



# For merged10_adapter:
print(f"Processing avg10adapters_task{cli_args.cluster_id}_phi-3...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=compute_dtype)
model = PeftModel.from_pretrained(model, f"dxltt1211/task{cli_args.cluster_id}_avgadapters")
model = model.merge_and_unload()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
pipe.tokenizer.apply_chat_template([{"role": "user", "content": chatml_datasets['test'][0]['messages'][0]['content']}], tokenize=False, add_generation_prompt=True)
metrics = chatml_datasets['test'].map(lambda row: calculate_rouge(pipe, row), batched=False)

rouge1_mean = np.mean(metrics['rouge1'])
rouge2_mean = np.mean(metrics['rouge2'])
rougel_mean = np.mean(metrics['rougeL'])
rougelsum_mean = np.mean(metrics['rougeLsum'])


output_file = os.path.join(output_dir, f"rouge_phi3_task{cli_args.cluster_id}_avg10adapters.txt")
with open(output_file, "w") as f:
    f.write(f"Avg Adapter - Rouge 1 Mean: {rouge1_mean}\n")
    f.write(f"Avg Adapter - Rouge 2 Mean: {rouge2_mean}\n")
    f.write(f"Avg Adapter - Rouge L Mean: {rougel_mean}\n")
    f.write(f"Avg Adapter - Rouge Lsum Mean: {rougelsum_mean}\n")

print(f"Results saved to {output_file}")
