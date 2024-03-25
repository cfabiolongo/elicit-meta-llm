import pandas as pd
import re

#lower bound
lb = 0.69

def contiene_numeri(stringa):
    # Utilizza espressione regolare per trovare tutti i numeri
    numeri = re.findall(r'\d+', stringa)
    
    # Se la lista di numeri non è vuota, la stringa contiene almeno un numero
    if numeri:
        return True
    else:
        return False
    
    
    
def confronta_numeri(stringa_a, stringa_b):
    # Funzione per estrarre i numeri da una stringa
    def estrai_numeri(stringa):
        return set(map(int, re.findall(r'\d+', stringa)))

    # Estrai i numeri dalle due stringhe
    numeri_a = estrai_numeri(stringa_a)
    numeri_b = estrai_numeri(stringa_b)

    # Controlla se ci sono numeri in comune
    numeri_comuni = numeri_a.intersection(numeri_b)

    # Restituisci True se ci sono numeri comuni, False altrimenti
    return bool(numeri_comuni)



def elimina_numeri_punto(stringa):
    # Utilizza espressione regolare per trovare tutti i numeri seguiti da un punto
    pattern = r'\b\d+\.\b'
    
    # Sostituisci tutte le occorrenze con una stringa vuota
    nuova_stringa = re.sub(pattern, '', stringa)
    
    return nuova_stringa




def check_admissibility(row):
    if row['Generated_meta']=="CORRECT" and row['Recall'] >= lb:
        a = contiene_numeri(row['Response'])
        b = contiene_numeri(row['Generated_Response'])
        print(f"\na: {row['Response']}, b: {row['Generated_Response']}")
        print(f"a: {b}, b: {b}")
        if a and b:
            c = confronta_numeri(row['Response'], row['Generated_Response'])
            print(f"c: {c}")
            if c:               
                return 'CORRECT'
            else:
                return 'WRONG'
        else:
            return 'CORRECT'
    elif row['Response'].lower() == row['Generated_Response'].lower():
        return 'CORRECT'
    elif row['Response'].lower() in row['Generated_Response'].lower(): 
        return 'CORRECT'
    elif row['Generated_Response'].lower() in row['Response'].lower():
        return 'CORRECT'     
    elif row['Precision'] > 0.83:
        return 'CORRECT'
    else:
        return 'WRONG'



# Leggi il file Excel
df = pd.read_excel('dataset/dolly_openqa_ultimepreds.xlsx')

# Applica la funzione alla colonna 'validazione' del DataFrame
# df['validazione'] = df['validazione'].apply(elimina_numeri_punto)

# Applica la funzione check_admissibility a ogni riga e crea la colonna "validation"
df['new_validation'] = df.apply(check_admissibility, axis=1)

# Salva il DataFrame modificato in un nuovo file Excel
df.to_excel('dataset/dolly_openqa_ultimeval2.xlsx', index=False)

print("File 'dolly_openqa_validations.xlsx' creato con successo con la colonna 'validation'.")

