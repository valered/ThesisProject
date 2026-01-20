import json
import os
import random

# Cartelle
RAW = "raw"        # dove hai i file V12 originali
OUT = "processed"  # dove salveremo i .txt.src / .txt.tgt

os.makedirs(OUT, exist_ok=True)

# -----------------------------
# Caricamento dei file V12
# -----------------------------
with open(os.path.join(RAW, "difftextV12.json"), encoding="utf-8") as f:
    diffs = json.load(f)

with open(os.path.join(RAW, "msgtextV12.json"), encoding="utf-8") as f:
    msgs = json.load(f)

# Controllo coerenza dimensioni
assert len(diffs) == len(msgs), "I due file hanno dimensioni diverse!"
N = len(diffs)
print(f"Numero di esempi totali: {N}")

# -----------------------------
# Accesso compatibile lista/dict
# -----------------------------
def get_item(container, i):
    """
    Gestisce sia il caso in cui il JSON sia una lista (container[i])
    sia il caso in cui sia un dict con chiavi '0','1',...
    """
    if isinstance(container, dict):
        return container[str(i)]
    else:
        return container[i]

# -----------------------------
# Pulizia testo (una sola riga)
# -----------------------------
def clean_text(x: str) -> str:
    """
    - Converte in stringa
    - Rimuove ritorni a capo interni (\r, \n)
    - Comprimi spazi multipli in uno
    - Strip finale
    """
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

# -----------------------------
# Creazione split train/valid/test
# -----------------------------
indices = list(range(N))
random.seed(42)
random.shuffle(indices)

# split 80-10-10
train_end = int(0.8 * N)
valid_end = int(0.9 * N)

train_idx = indices[:train_end]
valid_idx = indices[train_end:valid_end]
test_idx  = indices[valid_end:]

def write_split(name, idx_list):
    src_path = os.path.join(OUT, f"{name}.txt.src")
    tgt_path = os.path.join(OUT, f"{name}.txt.tgt")

    with open(src_path, "w", encoding="utf-8") as fs, \
         open(tgt_path, "w", encoding="utf-8") as ft:
        for i in idx_list:
            src_raw = get_item(diffs, i)
            tgt_raw = get_item(msgs, i)

            src = clean_text(src_raw)
            tgt = clean_text(tgt_raw)

            fs.write(src + "\n")
            ft.write(tgt + "\n")

    print(f"{name}: {len(idx_list)} esempi → {src_path}, {tgt_path}")

# -----------------------------
# Generazione dei file
# -----------------------------
write_split("train", train_idx)
write_split("valid", valid_idx)
write_split("test",  test_idx)

print("✅ Dataset CoDiSum V12 ricostruito correttamente!")
