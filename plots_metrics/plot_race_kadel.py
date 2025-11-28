import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================
# Path di progetto
# ================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RACE_CSV = os.path.join(
    PROJECT_ROOT, "RACE", "results_race", "race_50k_metrics_all_seeds.csv"
)
KADEL_CSV = os.path.join(
    PROJECT_ROOT, "KADEL", "results_kadel", "kadel_metrics_all_seeds.csv"
)

# I grafici verranno salvati nella stessa cartella dello script
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# Lettura dei dati
# ================================
def load_metrics(csv_path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["model"] = model_name
    return df


race_df = load_metrics(RACE_CSV, "RACE")
kadel_df = load_metrics(KADEL_CSV, "KADEL")

# Uniamo per comodità
all_df = pd.concat([race_df, kadel_df], ignore_index=True)

# Colonne di metriche che ci interessano (lista completa, se serve)
METRICS = [
    "BLEU",
    "METEOR",
    "ROUGE1",
    "ROUGE2",
    "ROUGEL",
    "SACREBLEU",
    "CodeBLEU_text",
    "CIDEr",
    "IDENTIFIER_RECALL",
]


# ================================
# Barplot media ± deviazione standard
# ================================
def plot_bar_mean_std(df: pd.DataFrame, metrics, output_path: str):
    """
    Crea un barplot con media ± deviazione standard per RACE vs KADEL
    per le metriche specificate.
    """
    # Calcolo media e std per ogni metrica e modello
    rows = []
    for metric in metrics:
        for model in ["RACE", "KADEL"]:
            vals = df.loc[df["model"] == model, metric].values
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "mean": np.mean(vals),
                    "std": np.std(vals, ddof=1),
                }
            )
    stat_df = pd.DataFrame(rows)

    metrics_order = metrics
    x = np.arange(len(metrics_order))  # posizioni delle metriche
    width = 0.35  # larghezza delle barre

    fig, ax = plt.subplots(figsize=(10, 4))

    # Dati per RACE
    race_means = [
        stat_df[(stat_df["metric"] == m) & (stat_df["model"] == "RACE")]["mean"].values[0]
        for m in metrics_order
    ]
    race_stds = [
        stat_df[(stat_df["metric"] == m) & (stat_df["model"] == "RACE")]["std"].values[0]
        for m in metrics_order
    ]

    # Dati per KADEL
    kadel_means = [
        stat_df[(stat_df["metric"] == m) & (stat_df["model"] == "KADEL")]["mean"].values[0]
        for m in metrics_order
    ]
    kadel_stds = [
        stat_df[(stat_df["metric"] == m) & (stat_df["model"] == "KADEL")]["std"].values[0]
        for m in metrics_order
    ]

    race_color = "#5a0010"
    kadel_color = "#166253"

    ax.bar(
        x - width / 2,
        race_means,
        width,
        yerr=race_stds,
        capsize=4,
        label="RACE",
        color=race_color,
    )
    ax.bar(
        x + width / 2,
        kadel_means,
        width,
        yerr=kadel_stds,
        capsize=4,
        label="KADEL",
        color=kadel_color,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_order, rotation=45, ha="right")
    ax.set_ylabel("Valore medio")
    # Niente titolo: lo metti nella caption in tesi
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ================================
# Boxplot per singola metrica
# ================================
def plot_boxplot_metric(df: pd.DataFrame, metric: str, output_path: str):
    """
    Crea un boxplot RACE vs KADEL per una singola metrica.
    """
    data = [
        df.loc[df["model"] == "RACE", metric].values,
        df.loc[df["model"] == "KADEL", metric].values,
    ]

    fig, ax = plt.subplots(figsize=(4, 5))

    bp = ax.boxplot(
        data,
        labels=["RACE", "KADEL"],
        patch_artist=True,
    )

    # Colori box
    bp["boxes"][0].set_facecolor("#5a0010")
    bp["boxes"][1].set_facecolor("#166253")

    # Mediane bianche
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(1.4)

    # (opzionale) bordi box in nero
    for box in bp["boxes"]:
        box.set_edgecolor("black")

    ax.set_ylabel(metric)
    # Niente titolo, lo spieghi nella caption
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    # 1) Barplot media ± std (SACREBLEU escluso per scala diversa)
    bar_metrics = ["BLEU", "METEOR", "ROUGEL", "CIDEr", "IDENTIFIER_RECALL"]
    bar_path = os.path.join(OUTPUT_DIR, "bar_mean_std_race_kadel.png")
    plot_bar_mean_std(all_df, bar_metrics, bar_path)
    print(f"Salvato barplot in: {bar_path}")

    # 2) Boxplot per ROUGEL
    box_rougel_path = os.path.join(OUTPUT_DIR, "boxplot_ROUGEL.png")
    plot_boxplot_metric(all_df, "ROUGEL", box_rougel_path)
    print(f"Salvato boxplot ROUGEL in: {box_rougel_path}")

    # 3) Boxplot per SACREBLEU (rimane il grafico dedicato)
    box_sacre_path = os.path.join(OUTPUT_DIR, "boxplot_SACREBLEU.png")
    plot_boxplot_metric(all_df, "SACREBLEU", box_sacre_path)
    print(f"Salvato boxplot SACREBLEU in: {box_sacre_path}")


if __name__ == "__main__":
    main()
