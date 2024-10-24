import pandas as pd

# Leggi i tre sheet del file Excel
file_path = 'dataset/squad_preds_metrics_deberta.xlsx'  # sostituisci con il percorso del tuo file
sheet1 = pd.read_excel(file_path, sheet_name='0.1')
sheet2 = pd.read_excel(file_path, sheet_name='0.6')
sheet3 = pd.read_excel(file_path, sheet_name='0.8')

# Unisci i tre sheet
combined_df = pd.concat([sheet1, sheet2, sheet3])

# Rimuovi i duplicati del campo 'Generated response', mantenendo la riga con il valore più alto in 'f1'
final_df = combined_df.sort_values('f1', ascending=False).drop_duplicates(subset='Generated_Response', keep='first')

# Scrivi il risultato in un nuovo file Excel o aggiungi un nuovo sheet
output_file = 'dataset/squad_merged_metrics_deberta.xlsx'  # sostituisci con il percorso del file di output
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    final_df.to_excel(writer, sheet_name='Unione', index=False)

print("File Excel creato con successo!")
