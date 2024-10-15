# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import statistics
from evaluate import load
from rouge import Rouge 



# Carica le metriche
bertscore = load("bertscore")
# rouge = load("rouge")
# bleu = load("bleu")

filename = 'gpt_preds_t0.1'

df = pd.read_excel(f'dataset/{filename}.xlsx')
dataset = Dataset.from_pandas(df)
print("#dataset items: ", len(dataset))

# Initialize lists to store metrics for each record
rouge_precisions = []
rouge_recalls = []
rouge_f1s = []
matches = []

# bleu_scores = []

match = 0
rouge = Rouge()

# Calcola BERTScore in modo aggregato
results_bertscore = bertscore.compute(predictions=dataset['Generated_Response'], references=dataset['response'], model_type="microsoft/deberta-v2-xxlarge-mnli")

precision_scores = results_bertscore['precision']
recall_scores = results_bertscore['recall']
f1_scores = results_bertscore['f1']

precision_media = statistics.mean(precision_scores)
recall_media = statistics.mean(recall_scores)
f1_media = statistics.mean(f1_scores)

print("\nMedia della precisione (BERTScore):", precision_media)
print("Media della recall (BERTScore):", recall_media)
print("Media della f1 (BERTScore):", f1_media)

for i in range(len(dataset)):

    print(f"\n---------------- Record #{i}:\n")
    
    d = dataset[i]

    rouge_score = rouge.get_scores(d['response'], d['Generated_Response'])[0]['rouge-l']
    print("\nrouge-l: ", rouge_score)

    # Aggiungi i punteggi ROUGE alle liste
    rouge_precisions.append(rouge_score['p'])
    rouge_recalls.append(rouge_score['r'])
    rouge_f1s.append(rouge_score['f'])

    if str(d['response']) == d['Generated_Response']:
        match = match + 1
        print("---> MATCH <---")
        matches.append("CORRECT")
    else:
        matches.append("???")

    # Calcola BLEU per questa predizione
    # bleu_result = bleu.compute(predictions=[gen], references=[[d['response']]])
    # bleu_scores.append(bleu_result['bleu'])

# writing excel dataframe with predictions (Generated_Response)
file_output = f"{filename}_validated"

df['Precision'] = precision_scores
df['Recall'] = recall_scores
df['f1'] = f1_scores

df['RG_L_P'] = rouge_precisions  # Aggiungi ROUGE precision per ogni record
df['RG_L_R'] = rouge_recalls        # Aggiungi ROUGE recall per ogni record
df['RG_L_F1'] = rouge_f1s              # Aggiungi ROUGE F1 per ogni record

# df['BLEU'] = bleu_scores  # Aggiungi BLEU per ogni record

df['VALIDATION'] = matches

print("\n#MATCH:", match)

df.to_excel(f"dataset/{file_output}.xlsx")

print(f"\nFile {file_output}.xlsx successfully created'.")
