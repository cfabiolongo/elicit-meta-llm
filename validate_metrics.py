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

filename = 'gpt_preds_t0.6'

df = pd.read_excel(f'dataset/{filename}.xlsx')
dataset = Dataset.from_pandas(df)
print("#dataset items: ", len(dataset))

# Initialize lists to store metrics for each record
rouge_precisions = []
rouge_recalls = []
rouge_f1s = []

bert_precisions = []
bert_recalls = []
bert_f1s = []

matches = []

# bleu_scores = []

match = 0
rouge = Rouge()

for i in range(len(dataset)):

    print(f"\n---------------- Record #{i}:\n")
    
    d = dataset[i]

    results_bert = bertscore.compute(predictions=[d['Generated_Response']], references=[d['response']], model_type="microsoft/deberta-v2-xxlarge-mnli")

    # BERTSCORE section
    precision_scores = results_bert['precision']
    recall_scores = results_bert['recall']
    f1_scores = results_bert['f1']

    bert_precisions.append(precision_scores)
    bert_recalls.append(recall_scores)
    bert_f1s.append(f1_scores)

    # ROUGE section
    rouge_score = rouge.get_scores(d['response'], d['Generated_Response'])[0]['rouge-l']
    print("\nrouge-l: ", rouge_score)

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

# Appiattisci le liste di precision, recall e f1
flat_bert_precisions = [item for sublist in bert_precisions for item in sublist]
flat_bert_recalls = [item for sublist in bert_recalls for item in sublist]
flat_bert_f1s = [item for sublist in bert_f1s for item in sublist]

# Calcola la media
precision_media = statistics.mean(flat_bert_precisions)
recall_media = statistics.mean(flat_bert_recalls)
f1_media = statistics.mean(flat_bert_f1s)

print("\nMedia della precisione (BERTScore):", precision_media)
print("Media della recall (BERTScore):", recall_media)
print("Media della f1 (BERTScore):", f1_media)


# writing excel dataframe with predictions (Generated_Response)
file_output = f"{filename}_validated"

df['Precision'] = flat_bert_precisions
df['Recall'] = flat_bert_recalls
df['f1'] = flat_bert_f1s

df['RG_L_P'] = rouge_precisions  # Aggiungi ROUGE precision per ogni record
df['RG_L_R'] = rouge_recalls        # Aggiungi ROUGE recall per ogni record
df['RG_L_F1'] = rouge_f1s              # Aggiungi ROUGE F1 per ogni record

# df['BLEU'] = bleu_scores  # Aggiungi BLEU per ogni record

df['VALIDATION'] = matches

print("\n#MATCH:", match)

df.to_excel(f"dataset/{file_output}.xlsx")

print(f"\nFile {file_output}.xlsx successfully created'.")
