import pandas as pd

# Carica i tre file Excel
file_01 = pd.read_excel('dataset/gpt_preds_metrics_t0.1_deberta.xlsx')
file_02 = pd.read_excel('dataset/gpt_preds_metrics_t0.6_deberta.xlsx')
file_03 = pd.read_excel('dataset/gpt_preds_metrics_t0.8_deberta.xlsx')

# Crea un oggetto ExcelWriter per il nuovo file
with pd.ExcelWriter('dataset/gpt_preds_metrics_deberta.xlsx', engine='xlsxwriter') as writer:
    # Scrivi i DataFrame nei rispettivi fogli (sheet)
    file_01.to_excel(writer, sheet_name='0.1', index=False)
    file_02.to_excel(writer, sheet_name='0.6', index=False)
    file_03.to_excel(writer, sheet_name='0.8', index=False)

print("File fabio.xlsx creato con successo!")