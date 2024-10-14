import pandas as pd
from transformers import AutoTokenizer
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rouge_score import rouge_scorer

import statistics
from evaluate import load
bertscore = load("bertscore")

base_model = "../models/7B-chat"
adapters_name1 = f"../models/finetuned/llama-metadolly_qa_super100_100ep"
adapters_name2 = f"../models/finetuned/llama-dollycontext3_qa_60ep"

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# https://huggingface.co/docs/peft/package_reference/lora

# adapters (list) — List of adapter names to be merged.
# weights (list) — List of weights for each adapter.
# adapter_name (str) — Name of the new adapter.
# combination_type (str) — Type of merging. Can be one of [svd, linear, cat]. When using the cat combination_type you should be aware that rank of the resulting adapter will be equal to the sum of all adapters ranks. So it’s possible that the mixed adapter may become too big and result in OOM errors.
# svd_rank (int, optional) — Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
# svd_clamp (float, optional) — A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform clamping. Defaults to None.
# svd_full_matrices (bool, optional) — Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned tensors U and Vh. Defaults to True.
# svd_driver (str, optional) — Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be one of [None, gesvd, gesvdj, gesvda]. For more info please refer to torch.linalg.svd documentation. Defaults to None.

# combination type
comb_type = "cat"
# combination weight [meta, dolly]
comb_weight = [0.5, 0.2]

Metavalidation = True
Autoregressive = True
Accept_last = True
Write_file_output = True

# Temperature value for for Meta-validation
meta_temp = 0.1
# Temperature value for the first qa prediction
first_temp = 0.1
# Temperature values for additional predictions for Meta-validation
temp_vect = [0.2, 0.3, 0.4, 0.5, 0.6]

print(f"\ncombination type: {comb_type}")
print(f"weights: {comb_weight}")
print(f"Meta-validation temperature: {meta_temp}")
print(f"QA initial temperature: {first_temp}")
print(f"QA additional temperatures vector: {temp_vect}")
print(f"Metavalidation: {Metavalidation}")
print(f"Autoregressive: {Autoregressive}\n")

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map={"": 0},  quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

model = PeftModel.from_pretrained(model, adapters_name2, adapter_name="dolly")
model.load_adapter(adapters_name1, adapter_name="meta")

model.add_weighted_adapter(["dolly", "meta"], comb_weight, combination_type=comb_type, adapter_name="dolly_meta")


model.delete_adapter("meta")
model.delete_adapter("dolly")
model.save_pretrained("../models/finetuned")

model = PeftModel.from_pretrained(model, "../models/finetuned/dolly_meta")

############################################# 
############ Meta-validator test ############
#############################################

df = pd.read_excel('dataset/dolly_openqa_validations.xlsx', sheet_name='Sheet1')

df = df.iloc[:100].reset_index(drop=True)  

question_column = df['Question'].tolist()
generated_column = df['Generated_Response'].tolist()
validation_column = df['validation'].tolist()
response_column = df['Response'].tolist()

# Create datasets
dataset = [{'validation': val, 'question': quest, 'generated': gen, 'response': resp} for val, quest, gen , resp in zip(validation_column, question_column, generated_column, response_column)]

meta_prompt="Validate the response given in Input with CORRECT or WRONG, considering the question given in Context."

preds = []
match = 0
correct_match = 0
count_correct = df[df['validation'] == 'CORRECT'].shape[0]

def query_Llama(context, prompt, input, temp, max_new_tokens):     
              
    prompt = f"""### Context:
    {context}
     ### Instruction:
    {prompt}
    ### Input:
    {input}

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
   
    return gen

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0


for i in range(len(dataset)):

    print(f"\n---------------- Record #{i}:\n")
    
    d = dataset[i] 
    
    gen = query_Llama(d['question'], meta_prompt, d['generated'], meta_temp, 5)               
               
    print(f"\nPrompt: {d['question']}\n")            
    print(f"Generated validation: {gen}")
    print(f"Ground truth: {d['validation']}\n")    

    preds.append(gen)  
                 
    if "CORRECT" in gen or "Correct" in gen:                
            if str(d['validation']) == "CORRECT":            
                true_positive = true_positive + 1           
            else:
                false_positive = false_positive + 1                                 
    else:                                 
                
        if str(d['validation']) == "CORRECT":            
            false_negative = false_negative + 1           
        else:
            true_negative = true_negative + 1                     
         
accuracy = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)
 
print(f"\n#Meta-validation scores:")
print(f"accuracy: {float(round(accuracy, 4))}")
print(f"precision: {float(round(precision, 4))}")
print(f"recall: {float(round(recall, 4))}")
print(f"f1-score: {float(round(f1, 4))}\n")




def attesa_input():
    # Stampa un messaggio
    print("\nPremi un tasto per continuare...")
    
    # Attendi l'input dell'utente
    input()

# Chiamata alla funzione di attesa
attesa_input()

# Il codice qui sotto verrà eseguito solo dopo che l'utente ha premuto un tasto
print("Esecuzione continua dopo l'input dell'utente...........")


###########################################################
############ Question-Answering+Metavalidation ############
###########################################################

print("\n\n########################################################\n\n")

sub_prompt="Generate a Response to the question given in Input. Response must be different from Context."

preds = []
meta_preds = []
rouge_precision = []
rouge_recall = []
rouge_f1 = []
match = 0

for i in range(len(dataset)):

    print(f"\n---------------- Record #{i}:\n")
    
    d = dataset[i]
    
    print(f"Prompt: {d['question']}\n")   
    
    gen = query_Llama("", sub_prompt, d['question'], first_temp, 50)
    print(f"First-{first_temp}-gen: {gen}")  
        
    if Metavalidation:        
        for i in range(len(temp_vect)): 
            meta_gen = query_Llama(d['question'], meta_prompt, gen, meta_temp, 5)
            print(f"Meta-validation: {meta_gen}")            
    
            if "CORRECT" in meta_gen or "Correct" in meta_gen or "correct" in meta_gen:                                                                    
                break
            else:
                print(f"Temp: {temp_vect[i]}\n")  
                if Autoregressive:
                    gen = query_Llama(gen, sub_prompt, d['question'], temp_vect[i], 50)
                else:
                    gen = query_Llama("", sub_prompt, d['question'], temp_vect[i], 50)                                    
                
            if i==len(temp_vect)-1:
                if Accept_last:
                    print(f"\nAttempts on temperature variation exhausted. Accepting last prediction with temp={temp_vect[i]}\n") 
                else: 
                    gen = "Unknown"                                                                    
                                 
    print(f"\nGenerated instruction:\n{gen}")
    print(f"Ground truth:\n{d['response']}\n")
    
    preds.append(gen)

    rg = scorer.score(d['response'], gen)
    print(rg)

    rouge_precision.append(str(rg['rouge1']).split(',')[0].split("=")[1])
    rouge_recall.append(str(rg['rouge1']).split(',')[1].split("=")[1])
    rouge_f1.append(str(rg['rouge1']).split(',')[2].split("=")[1][:-1])

      
    if Metavalidation:
    	meta_preds.append(meta_gen)       
    
    if str(d['response']).lower() == gen.lower():
        match = match + 1
        print("---> MATCH <---")     
        
print(f"\n#Match: {match}\n")
    
# distilbert-base-uncased, roberta-large, bert-large-uncased, deberta-large, deberta-xlarge, deberta-xlarge-mnli
results = bertscore.compute(predictions=preds, references=df['Response'], model_type="microsoft/deberta-large")

# Estrai la precisione dai risultati
precision_scores = results['precision']
# Estrai la precisione dai risultati
recall_scores = results['recall']
# Estrai la precisione dai risultati
f1_scores = results['f1']

precision_media = statistics.mean(precision_scores)
print("Media della precisione:", precision_media)

recall_media = statistics.mean(recall_scores)
print("Media della recall:", recall_media)

f1_media = statistics.mean(f1_scores)
print("Media della f1:", f1_media)


if Write_file_output and Metavalidation:
    
    # Creazione del nuovo DataFrame con solo le colonne 'Question' e 'Response'
    new_df = pd.DataFrame({'Question': [d['question'] for d in dataset], 'Response': [d['response'] for d in dataset]})

    new_df['Generated_Response'] = preds 
    new_df['Generated_meta'] = meta_preds
    new_df['Precision'] = precision_scores 
    new_df['Recall'] = recall_scores  
    new_df['F1'] = f1_scores

    new_df['Precision rouge'] = rouge_precision
    new_df['Recall rouge'] = rouge_recall 
    new_df['F1 rouge'] = rouge_f1

    # Definire il percorso del file Excel
    excel_file = 'dataset/dolly_openqa_validations.xlsx'

    # Scrivere il DataFrame nel file Excel
    # new_df.to_excel(excel_file, sheet_name='Sheet2', index=False)

    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
           new_df.to_excel(writer, sheet_name='Sheet2-'+comb_type, index=False)

    print("File Excel creato con successo:", excel_file)