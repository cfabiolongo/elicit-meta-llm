# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import statistics
from evaluate import load
from rouge import Rouge 



# Carica le metriche
bertscore = load("bertscore")
# bleu = load("bleu")

# General parameters
model_name = "llama-dolly_qa_100ep"
output_dir = f"/home/fabio/llama/models/finetuned/{model_name}"
temp = 0.8
max_new_tokens = 512
gpu = "cuda"

print(f"\nModel: {output_dir}\nTemperature: {temp}\n")

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,    
    load_in_4bit=True,
    device_map=gpu,        
)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Load dataset from the hub
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Filtra il dataset
filtered_dataset = dataset.filter(lambda example: example['category'] == "open_qa" and len(example["response"]) <= 100)
dataset = filtered_dataset.select(range(100))

print("#dataset items: ", len(dataset))

sub_prompt="Generate a response to the question given in Input."

preds = []
match = 0

# Initialize lists to store metrics for each record
rouge_precisions = []
rouge_recalls = []
rouge_f1s = []

# bleu_scores = []

rouge = Rouge()

for i in range(len(dataset)):

    print(f"\n---------------- Record #{i}:\n")
    
    d = dataset[i]   
              
    prompt = f"""### Instruction:
    {sub_prompt}   
    ### Input:
    {d['instruction']}

    ### Response:
    """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    
    outputs = model.generate(
        input_ids=input_ids,        
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=temp,
        pad_token_id=model.config.eos_token_id,  # Imposta pad_token_id su eos_token_id
        attention_mask=torch.ones_like(input_ids)  # Imposta l'attention mask
    )
    
    gen_full = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    gen = gen_full.split("#")[0]
    gen = gen.strip()
            
    print(f"\nPrompt: {d['instruction']}\n")
    print(f"Generated instruction:\n{gen}")
    
    
    print(f"Ground truth:\n{d['response']}")

    preds.append(gen)       
    
    if str(d['response']).lower() == gen.lower():
        match = match + 1
        print("---> MATCH <---")       
      
    reference = gen if not len(gen)==0 else "empty"
    
    rouge_score = rouge.get_scores(d['response'], reference)[0]['rouge-l']    
    
    print("\nrouge-l: ", rouge_score)
       
    # Aggiungi i punteggi ROUGE alle liste
    rouge_precisions.append(rouge_score['p'])
    rouge_recalls.append(rouge_score['r'])
    rouge_f1s.append(rouge_score['f'])
    
    # Calcola BLEU per questa predizione
    # bleu_result = bleu.compute(predictions=[gen], references=[[d['response']]])
    # bleu_scores.append(bleu_result['bleu'])

# Calcola BERTScore in modo aggregato
results_bertscore = bertscore.compute(predictions=preds, references=dataset['response'], model_type="microsoft/deberta-v2-xxlarge-mnli")

precision_scores = results_bertscore['precision']
recall_scores = results_bertscore['recall']
f1_scores = results_bertscore['f1']

precision_media = statistics.mean(precision_scores)
recall_media = statistics.mean(recall_scores)
f1_media = statistics.mean(f1_scores)

print("\n#MATCH:", match)

print("\nMedia della precisione (BERTScore):", precision_media)
print("Media della recall (BERTScore):", recall_media)
print("Media della f1 (BERTScore):", f1_media)

# writing excel dataframe with predictions (Generated_Response)
filename = f"dolly_preds_metrics_t{temp}"

# Create the pandas DataFrame
df = pd.DataFrame()

df['instruction'] = dataset['instruction']
df['response'] = dataset['response']

df['Generated_Response'] = preds
df['Precision'] = precision_scores
df['Recall'] = recall_scores
df['f1'] = f1_scores

df['RG_L_P'] = rouge_precisions  # Aggiungi ROUGE precision per ogni record
df['RG_L_R'] = rouge_recalls        # Aggiungi ROUGE recall per ogni record
df['RG_L_F1'] = rouge_f1s              # Aggiungi ROUGE F1 per ogni record

# df['BLEU'] = bleu_scores  # Aggiungi BLEU per ogni record

df.to_excel(f"dataset/{filename}.xlsx")

print(f"\nFile {filename}.xlsx successfully created with column 'Generated_Response'.")
