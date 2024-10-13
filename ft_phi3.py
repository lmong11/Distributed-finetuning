import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, set_seed
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
import time
import tracemalloc
import psutil
import numpy as np
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# Set the global parameters
model_id = "microsoft/Phi-3-mini-4k-instruct"
dataset_name = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
dataset_split = "train"
new_model = "avgadapters_23_phi-3"
hf_model_repo = "lmong/" + new_model
device_map = {"": 0}
use_4bit = True
bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_4bit_quant_type = "nf4"
use_double_quant = True
lora_r, lora_alpha, lora_dropout = 16, 16, 0.05
target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

set_seed(34)
notebook_login()

# Load dataset
dataset = load_dataset(dataset_name, split=dataset_split).train_test_split(test_size=0.99, seed=34)['train']
print(f"Dataset size: {len(dataset)}")

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

dataset_chatml = dataset.map(create_message_column).map(format_dataset_chatml).train_test_split(test_size=0.05, seed=12)

# Load model with quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=use_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type,
                                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype, bnb_4bit_use_double_quant=use_double_quant)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=bnb_4bit_compute_dtype, quantization_config=bnb_config, 
                                             device_map=device_map, trust_remote_code=True)

model = prepare_model_for_kbit_training(model)
args = TrainingArguments(output_dir="./phi-3-mini-QLoRA/Adapter3/", evaluation_strategy="steps", do_eval=True,
                         optim="adamw_torch", per_device_train_batch_size=16, per_device_eval_batch_size=16,
                         gradient_accumulation_steps=4, logging_steps=100, learning_rate=1e-4, num_train_epochs=3,
                         warmup_ratio=0.1, lr_scheduler_type="linear", report_to="wandb", seed=42)

peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type=TaskType.CAUSAL_LM, 
                         target_modules=target_modules)

# Initialize W&B
wandb.init(project="phi3", name="phi3")

# Trainer
trainer = SFTTrainer(model=model, train_dataset=dataset_chatml['train'], eval_dataset=dataset_chatml['test'], 
                     peft_config=peft_config, dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer, args=args)

# Memory profiling
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # in MB

tracemalloc.start()
start_time, start_memory = time.time(), get_memory_usage()
print(f"Initial memory usage: {start_memory:.2f} MB")

# Train and save model
trainer.train()
trainer.save_model()

end_time, end_memory = time.time(), get_memory_usage()
print(f"Training completed in: {end_time - start_time:.2f} seconds")
print(f"Final memory usage: {end_memory:.2f} MB")
_, peak = tracemalloc.get_traced_memory()
print(f"Peak memory usage during training: {peak / 1024 ** 2:.2f} MB")
tracemalloc.stop()

# Push model to Hugging Face Hub
trainer.push_to_hub(hf_model_repo)

# Merge Adapters
def merge_adapters(adapter1_repo, adapter2_repo, model):
    adapter1 = PeftModel.from_pretrained(model, adapter1_repo)
    adapter2 = PeftModel.from_pretrained(model, adapter2_repo)
    avg_weights = {}
    for (name1, param1), (name2, param2) in zip(adapter1.named_parameters(), adapter2.named_parameters()):
        if name1 == name2:
            avg_weights[name1] = (param1.data + param2.data) / 2.0
    for name, param in adapter1.named_parameters():
        if name in avg_weights:
            param.data.copy_(avg_weights[name])
    return adapter1

merged_adapter = merge_adapters("lmong/Adapter2", "lmong/Adapter3", model)
merged_adapter.save_pretrained("dxltt1211/Adapter23_avg")

# Evaluate model
def test_inference(prompt, model, tokenizer):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return pipe(prompt, max_new_tokens=256)[0]['generated_text'][len(prompt):].strip()

rouge_metric = evaluate.load("rouge")
def calculate_rogue(row, model, tokenizer):
    response = test_inference(row['messages'][0]['content'], model, tokenizer)
    return rouge_metric.compute(predictions=[response], references=[row['response']])

# Inference and memory profiling
start_time, start_memory = time.time(), get_memory_usage()
print(f"Initial memory usage: {start_memory:.2f} MB")

metricas = dataset_chatml['test'].select(range(0, 102)).map(lambda row: calculate_rogue(row, model, tokenizer))
end_time, end_memory = time.time(), get_memory_usage()
print(f"Inference completed in: {end_time - start_time:.2f} seconds")
print(f"Final memory usage: {end_memory:.2f} MB")

# Save Rouge results
with open("rouge_scores_phi3_adapter3.txt", "w") as f:
    f.write(f"Rouge 1 Mean: {np.mean(metricas['rouge1']):.2f}\n")
    f.write(f"Rouge 2 Mean: {np.mean(metricas['rouge2']):.2f}\n")
    f.write(f"Rouge L Mean: {np.mean(metricas['rougeL']):.2f}\n")
    f.write(f"Rouge Lsum Mean: {np.mean(metricas['rougeLsum']):.2f}\n")

# Rouge score comparison
file1, file2, file3 = "rouge_scores_avgadapters_23_phi3.txt", "rouge_scores_phi3_adapter2.txt", "rouge_scores_phi3_adapter3.txt"
def read_scores(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        return {line.split(':')[0]: float(line.split(':')[1].strip()) for line in lines}

df = pd.DataFrame({
    "Avg Adapters": read_scores(file1),
    "Adapter 2": read_scores(file2),
    "Adapter 3": read_scores(file3)
})
df.to_excel("rouge_scores_comparison.xlsx")
