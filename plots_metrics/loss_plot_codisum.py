import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Cartella base
script_dir = Path(__file__).resolve().parent
base_path = script_dir.parent / "CoDiSum" / "results"
seeds = ["seed42", "seed123", "seed999"]
colors = ["#166253", "#5a0010", "#a3475c"]  # Verde, bordeaux, sfumatura

plt.figure(figsize=(9, 5))

for seed, color in zip(seeds, colors):
    file_path = base_path / f"run_{seed}" / "train_log.csv"
    if not file_path.exists():
        print(f"[Errore] File non trovato: {file_path}")
        continue
    df = pd.read_csv(file_path)

    plt.plot(df["epoch"], df["loss"], label=f"Train – {seed}", color=color, linewidth=2, linestyle='-')
    plt.plot(df["epoch"], df["val_loss"], label=f"Val – {seed}", color=color, linewidth=2, linestyle='--')

plt.title("Andamento della Training e Validation Loss – CoDiSum")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvataggio
output_path = script_dir / "loss_codisum.png"
plt.savefig(output_path, dpi=300)
plt.show()
