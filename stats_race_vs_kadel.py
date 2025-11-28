import os
import math
import numpy as np
import pandas as pd
from scipy import stats

# ===========
# PATH
# ===========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RACE_CSV = os.path.join(ROOT_DIR, "RACE", "results_race", "race_50k_metrics_all_seeds.csv")
KADEL_CSV = os.path.join(ROOT_DIR, "KADEL", "results_kadel", "kadel_metrics_all_seeds.csv")

OUT_CSV  = os.path.join(ROOT_DIR, "stats_race_vs_kadel.csv")

# ===========
# METRICHE DA ANALIZZARE
# (usa nomi "canonici" indipendenti da maiuscole/minuscole)
# ===========
METRICS = [
    "BLEU",
    "METEOR",
    "ROUGE1",
    "ROUGE2",
    "ROUGEL",
    "SACREBLEU",
    "IDENTIFIER_RECALL",
    "CodeBLEU_text",
    "CIDEr",
]

# Mappo ogni metrica alle possibili varianti di nome nei CSV
CANDIDATE_COLS = {
    "BLEU": ["BLEU", "bleu"],
    "METEOR": ["METEOR", "meteor"],
    "ROUGE1": ["ROUGE1", "rouge1"],
    "ROUGE2": ["ROUGE2", "rouge2"],
    "ROUGEL": ["ROUGEL", "rougeL", "rougel"],
    "SACREBLEU": ["SACREBLEU", "sacrebleu"],
    "IDENTIFIER_RECALL": ["IDENTIFIER_RECALL", "identifier_recall"],
    "CodeBLEU_text": ["CodeBLEU_text", "codebleu_text", "codebleu"],
    "CIDEr": ["CIDEr", "cider", "CIDER"],
}

def get_col(df: pd.DataFrame, metric: str):
    """Trova la colonna giusta nel DataFrame per una metrica."""
    candidates = CANDIDATE_COLS.get(metric, [metric])
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float)
    raise KeyError(f"Nessuna colonna trovata per la metrica {metric} in {df.columns}")


def cohen_d(x, y, equal_var=True):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    nx = len(x)
    ny = len(y)
    mx = x.mean()
    my = y.mean()
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)

    if equal_var:
        sp = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
        if sp == 0:
            return 0.0
        return (mx - my) / sp
    else:
        # versione "Glass's delta" semplificata: uso sd di RACE come riferimento
        s_ref = sx if sx > 0 else 1e-8
        return (mx - my) / s_ref


def main():
    if not os.path.exists(RACE_CSV):
        raise FileNotFoundError(f"CSV RACE non trovato: {RACE_CSV}")
    if not os.path.exists(KADEL_CSV):
        raise FileNotFoundError(f"CSV KADEL non trovato: {KADEL_CSV}")

    race = pd.read_csv(RACE_CSV)
    kadel = pd.read_csv(KADEL_CSV)

    print("RACE CSV columns:", race.columns.tolist())
    print("KADEL CSV columns:", kadel.columns.tolist())

    rows = []

    for metric in METRICS:
        try:
            race_vals = get_col(race, metric).dropna().values
            kadel_vals = get_col(kadel, metric).dropna().values
        except KeyError as e:
            print(f"\n⚠️ Metrica {metric} non trovata in uno dei CSV, la salto. Dettagli: {e}")
            continue

        if len(race_vals) < 2 or len(kadel_vals) < 2:
            print(f"\n⚠️ Metrica {metric}: troppi pochi valori per fare test statistici, salto.")
            continue

        print(f"\n===== {metric} =====")
        print("RACE:", race_vals)
        print("KADEL:", kadel_vals)

        mean_r = float(np.mean(race_vals))
        std_r  = float(np.std(race_vals, ddof=1))
        mean_k = float(np.mean(kadel_vals))
        std_k  = float(np.std(kadel_vals, ddof=1))

        # Shapiro–Wilk (attenzione: n=3 → interpretazione molto debole)
        try:
            sh_r = stats.shapiro(race_vals)
            sh_k = stats.shapiro(kadel_vals)
            shapiro_p_r = float(sh_r.pvalue)
            shapiro_p_k = float(sh_k.pvalue)
        except Exception as e:
            print(f"  Shapiro fallito per {metric}: {e}")
            shapiro_p_r = np.nan
            shapiro_p_k = np.nan

        # Levene (omogeneità delle varianze)
        try:
            lev = stats.levene(race_vals, kadel_vals)
            levene_p = float(lev.pvalue)
        except Exception as e:
            print(f"  Levene fallito per {metric}: {e}")
            levene_p = np.nan

        equal_var = not (levene_p < 0.05)

        # t-test (RACE vs KADEL)
        t_res = stats.ttest_ind(race_vals, kadel_vals, equal_var=equal_var)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

        # gradi di libertà (uso formula Welch se equal_var=False)
        nx = len(race_vals)
        ny = len(kadel_vals)
        sx2 = std_r**2
        sy2 = std_k**2

        if equal_var:
            df = nx + ny - 2
        else:
            num = (sx2 / nx + sy2 / ny) ** 2
            den = (sx2**2 / (nx**2 * (nx - 1))) + (sy2**2 / (ny**2 * (ny - 1)))
            df = num / den if den > 0 else nx + ny - 2

        # Cohen's d
        d = float(cohen_d(race_vals, kadel_vals, equal_var=equal_var))

        # Intervallo di confidenza 95% sulla differenza delle medie (RACE - KADEL)
        diff = mean_r - mean_k
        se_diff = math.sqrt(sx2 / nx + sy2 / ny)
        try:
            t_crit = stats.t.ppf(0.975, df)
            ci_low = diff + (-t_crit * se_diff)
            ci_high = diff + (t_crit * se_diff)
        except Exception:
            ci_low = np.nan
            ci_high = np.nan

        print(f"  mean_RACE = {mean_r:.4f}  (std={std_r:.4f})")
        print(f"  mean_KADEL = {mean_k:.4f} (std={std_k:.4f})")
        print(f"  diff (RACE - KADEL) = {diff:.4f}")
        print(f"  Shapiro p (RACE) = {shapiro_p_r:.4f}, (KADEL) = {shapiro_p_k:.4f}")
        print(f"  Levene p = {levene_p:.4f}  -> equal_var = {equal_var}")
        print(f"  t({df:.2f}) = {t_stat:.4f}, p = {t_p:.4f}")
        print(f"  Cohen's d = {d:.4f}")
        print(f"  95% CI diff = [{ci_low:.4f}, {ci_high:.4f}]")

        rows.append({
            "metric": metric,
            "mean_RACE": mean_r,
            "std_RACE": std_r,
            "mean_KADEL": mean_k,
            "std_KADEL": std_k,
            "diff_RACE_minus_KADEL": diff,
            "shapiro_p_RACE": shapiro_p_r,
            "shapiro_p_KADEL": shapiro_p_k,
            "levene_p": levene_p,
            "equal_var": equal_var,
            "t_stat": t_stat,
            "t_pvalue": t_p,
            "df": df,
            "cohen_d": d,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })

    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(OUT_CSV, index=False)
        print(f"\n✅ Risultati statistici salvati in: {OUT_CSV}")
    else:
        print("\n❌ Nessuna metrica analizzata (controlla i CSV).")

if __name__ == "__main__":
    main()
