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
path_model = f"../models/finetuned/llama2-dollyexar_{epoche}ep"

####################################### from dataset

# Load dataset from the hub
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Filtra il dataset
filtered_dataset = dataset.filter(lambda example: example['category'] == "open_qa" and len(example["response"]) <= 100 and example["context"] == "")

# Reduce dataset to the first 1000 records
dataset1 = filtered_dataset.select(range(1000))

df1 = dataset1.to_pandas()
df1 = df1[['instruction', 'response']]
df1['generated'] = ""
df1['validation'] = ""


####################################### from past inference

df2 = pd.read_excel('dataset/dolly_merged_metrics_deberta.xlsx')

df2['response'] = df2['Generated_Response'].values
df2 = df2.rename(columns={'Question': 'instruction'})
df2 = df2.rename(columns={'Generated_Response': 'generated'})
df2 = df2.rename(columns={'VALIDATION': 'validation'})
df2 = df2[['instruction', 'response', 'generated', 'validation']]

# Concateniamo i due DataFrame
df_conc = pd.concat([df1, df2])

# dataset_conc = Dataset.from_pandas(df_conc)
dataset_val = Dataset.from_pandas(df2)

print(f"dataset size: {len(dataset_val)}")
print(dataset_val[randrange(len(dataset_val))])

####################################### 

sub_prompt="Generate a Response to the question given in Input. Response must be different from Context."


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







