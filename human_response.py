import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import json
import argparse
from human_eval.evaluation import evaluate_functional_correctness
import re
# Set up command-line arguments
parser = argparse.ArgumentParser(description="Evaluate a model on HumanEval with modified prompts")
parser.add_argument(
    "--model",
    choices=["base", "finetuned"],
    required=True,
    help="Specify which model to load: 'base' or 'finetuned'"
)
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(0)
device_map = {"": 0}

# Define model paths
base_model_path = "microsoft/Phi-3-mini-4k-instruct"
finetuned_model_path = "./dxltt1211/task0_singleadapter"  # Replace with your fine-tuned model path

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load selected model based on the command-line argument
if args.model == "base":
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    sample_file = "generated_samples_base_parsed_92.jsonl"
elif args.model == "finetuned":
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    sample_file = "generated_samples_finetuned_parsed.jsonl"

# Load the modified HumanEval dataset
print("Loading HumanEval dataset with instructions...")
dataset = load_dataset("lmong/human-eval-instructions")

# Function to generate a response
def generate_response(model, tokenizer, prompt):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generation_args = {
        "max_new_tokens": 600,
        "return_full_text": False,
        "temperature": 0.3,
        "do_sample": False,
    }
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

import re

def extract_executable_code(response):
    """
    Extracts only the executable Python code from a model's response while preserving indentation.
    
    Parameters:
    response (str): The model's full response containing reasoning, explanations, and code.

    Returns:
    str: The filtered Python code containing only the executable parts, preserving indentation.
    """
    # Use a regex pattern to match Python code blocks
    code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
    
    # If no Python code blocks are found, return an empty string
    if not code_blocks:
        print("No Python code blocks found in the response.")
        return ""
    
    # Extract the first code block
    executable_code = code_blocks[0].strip()
    print("Extracted code block:", executable_code)  # Debugging extracted block

    # Remove unnecessary comments and docstrings
    executable_code = re.sub(r'""".*?"""', '', executable_code, flags=re.DOTALL)  # Remove triple-quoted docstrings
    executable_code = re.sub(r"#.*", '', executable_code)  # Remove inline comments
    
    # Split lines and keep valid ones while preserving indentation
    valid_code_lines = []
    for line in executable_code.splitlines():
        # Include lines that are not empty
        if line.strip():
            valid_code_lines.append(line)
    
    # Join the filtered lines to reconstruct the executable code
    filtered_code = "\n".join(valid_code_lines)
    print("Filtered valid code lines before formatting:", valid_code_lines)  # Debugging filtered lines

    # Validate the syntax of the filtered code
    try:
        compile(filtered_code, "<string>", "exec")  # Compile to check for syntax/indentation errors
    except IndentationError as e:
        print(f"Indentation Error: {e}")
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return ""  # Return an empty string if syntax errors exist
    
    return filtered_code

# Generate responses for the specified model using the 'instruction' column
print(f"Generating responses for {args.model} model...")
generated_samples = []
task_id_to_test = 1  # Specify the task ID you want to test
for i, example in enumerate(dataset['test']):
    if i == task_id_to_test: 
        task_id = example["task_id"]
        instruction_prompt = example['instruction']  # Use the 'instruction' column as the prompt
        print("Instruction Prompt:", instruction_prompt)
        # Generate response from the specified model
        response = generate_response(model, tokenizer, instruction_prompt)
        print("Original Response:",response)
        ### format the response"
        # response = response.split('```')[1] ### first we take the python code out
        # response = ":".join(response.split(':')[1:]) ### then we take the function definition out
        # response = response.split('"""')[-1] ### then we take the function description out
        response = extract_executable_code(response)
        print("Parsed Response:",response)
        # Append to generated samples with instruction prompt and response
        generated_samples.append({
            "task_id": task_id,
            "instruction_prompt": instruction_prompt,  # Save the instruction prompt
            "completion": response                     # Save the model's response
        })
        break

# Save generated samples for HumanEval to process
print(f"Saving generated samples to {sample_file}...")
with open(sample_file, "w") as f:
    for sample in generated_samples:
        f.write(json.dumps(sample) + "\n")

# Run the evaluation for the specified model
print(f"Evaluating {args.model} model...")
try:
    pass_at_k = evaluate_functional_correctness(sample_file)
    print(f"HumanEval pass@k results for {args.model} model: {pass_at_k}")
except AssertionError as e:
    print(f"Error during {args.model} model evaluation: {e}")
    print("Some problems are not attempted in the generated samples.")
