import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch


# General parameters
model_name = "llama2-metagpt_ep70"
output_dir = f"../models/finetuned/{model_name}"
temp = 0.6
max_new_tokens = 512
gpu = "cuda:0"

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

df = pd.read_excel('dataset/gpt_merged_metrics_deberta.xlsx')

df = df.iloc[:100].reset_index(drop=True)  

question_column = df['Question'].tolist()
generated_column = df['Generated_Response'].tolist()
validation_column = df['VALIDATION'].tolist()

# Create datasets
dataset = [{'validation': val, 'question': q, 'generated': g} for val, q, g in zip(validation_column, question_column, generated_column)]

dataset = dataset[:100]

sub_prompt="Validate the response given in Input with CORRECT or WRONG, considering the question given in Context."

preds = []
match = 0
correct_match = 0

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

for i in range(len(dataset)):

    print(f"\n---------------- Record #{i}:\n")
    
    d = dataset[i]   
              
    prompt = f"""### Context:
    {d['question']}
     ### Instruction:
    {sub_prompt}
    ### Input:
    {d['generated']}

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
            
    print(f"\nPrompt: {d['question']}\n")            
    print(f"Generated validation:\n{gen}")
    print(f"Ground truth validation:\n{d['VALIDATION']}")

    preds.append(gen)
    
    if "CORRECT" in gen or "Correct" in gen or "correct" in gen:                
            if str(d['VALIDATION']) == "CORRECT":
                true_positive = true_positive + 1           
            else:
                false_positive = false_positive + 1                                 
    else:                                 
                
        if str(d['VALIDATION']) == "CORRECT":
            false_negative = false_negative + 1           
        else:
            true_negative = true_negative + 1 
            
    if str(d['VALIDATION']).lower() == gen.lower():
        match = match + 1
        print("---> MATCH <---")
        
print(f"\n#Match: {match}\n")   
        
accuracy = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)
 
print(f"\n#Meta-validation scores:")
print(f"accuracy: {float(round(accuracy, 4))}")
print(f"precision: {float(round(precision, 4))}")
print(f"recall: {float(round(recall, 4))}")
print(f"f1-score: {float(round(f1, 4))}\n")
