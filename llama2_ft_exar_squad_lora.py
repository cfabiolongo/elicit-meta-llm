import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset, Dataset
from random import randrange
import pandas as pd

# General parameters
epoche = 60
lr = 2e-3
path_model = f"../models/finetuned/llama2-gptexar_{epoche}ep"


####################################### from past inference

df = pd.read_excel('dataset/squad_merged_metrics_deberta.xlsx')

df = df.rename(columns={'Generated_Response': 'generated'})
df = df.rename(columns={'VALIDATION': 'validation'})
df = df.rename(columns={'answer_text': 'response'})
df = df.rename(columns={'question': 'instruction'})

df = df[['instruction', 'response', 'generated', 'validation']]

dataset_val = Dataset.from_pandas(df)

print(f"dataset size: {len(dataset_val)}")
print(dataset_val[randrange(len(dataset_val))])

####################################### 

sub_prompt = "Generate a Response to the question given in Input. Response must be different from Context."


def format_instruction(sample):
    if sample['validation'] == "CORRECT":
        return f"""### Context: ### Instruction: {sub_prompt} ### Input: {sample['instruction']} ### Response: {sample['response']}"""    
    else:
        return f"""### Context: {sample['generated']} ### Instruction: {sub_prompt} ### Input: {sample['instruction']} ### Response: {sample['response']} """

print(format_instruction(dataset_val[randrange(len(dataset_val))]))

# Hugging Face model id
# model_id = "../models/7B"  # non-gated
model_id = "../models/7B-chat"  # non-gated

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,        
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

use_flash_attention = False

args = TrainingArguments(
    output_dir=path_model,
    num_train_epochs=epoche,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=lr,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm= False # disable tqdm since with packing values are in correct
)


from trl import SFTTrainer

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_val,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,  
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model(path_model)

print("\nFinetuning complete.")
print(f"\nPath model: {path_model}\n")







