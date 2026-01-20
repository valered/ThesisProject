import os
import json
import random
import re

import numpy as np
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
import pandas as pd

# CIDEr (opzionale)
try:
    from pycocoevalcap.cider.cider import Cider
    HAS_CIDER = True
except Exception:
    HAS_CIDER = False


# =========================
# PATH / CONFIG
# =========================

# cartella principale del progetto RACE (questo file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dove hai i risultati delle run
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# dove salviamo le metriche
RESULTS_DIR = os.path.join(BASE_DIR, "results_race")
os.makedirs(RESULTS_DIR, exist_ok=True)

# nomi delle cartelle per i tre seed
SEEDS = [42, 123, 999]
RUN_DIR_TPL = "RACE_50k_seed{seed}"


# =========================
# UTILS BASE
# =========================

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


# =========================
# IDENTIFIER RECALL
# =========================

def is_identifier(tok: str) -> bool:
    """Stessa logica dello script di CoDiSum: token che 'somigliano' a identificatori."""
    if not tok:
        return False
    # deve contenere almeno una lettera
    if not re.search(r"[A-Za-z]", tok):
        return False
    # pattern tipici: numeri, underscore, camelCase/PascalCase
    return bool(
        re.search(r"\d", tok)
        or "_" in tok
        or re.search(r"[A-Z]", tok[1:])
    )


def identifier_recall(preds, refs):
    recalls = []
    for p, r in zip(preds, refs):
        ids_ref = {t for t in r.split() if is_identifier(t)}
        if not ids_ref:
            continue
        ids_pred = {t for t in p.split() if is_identifier(t)}
        if not ids_pred:
            recalls.append(0.0)
        else:
            recalls.append(len(ids_ref & ids_pred) / len(ids_ref))
    return float(np.mean(recalls)) if recalls else 0.0


# =========================
# CODEBLEU_TEXT E CIDEr
# =========================

def compute_cider(preds, refs):
    """CIDEr via pycocoevalcap; se la libreria manca, restituisce None."""
    if not HAS_CIDER:
        return None
    gts = {i: [refs[i]] for i in range(len(refs))}
    res = {i: [preds[i]] for i in range(len(preds))}
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return float(score)


def compute_codebleu_text(preds, refs):
    """
    Versione 'testuale' in stile CoDiSum:
    pesi lessicali in base alla frequenza e F1 pesato.
    """
    freq = {}
    for txt in list(preds) + list(refs):
        for tok in txt.split():
            freq[tok] = freq.get(tok, 0) + 1

    def token_weight(tok):
        f = freq.get(tok, 1)
        return 1.0 / np.log2(f + 1.5)

    scores = []
    for p, r in zip(preds, refs):
        p_toks = p.split()
        r_toks = r.split()
        if not r_toks:
            continue

        # recall lato reference
        w_r = {i: token_weight(tok) for i, tok in enumerate(r_toks)}
        r_weight_total = sum(w_r.values())
        r_match = sum(w_r[i] for i, tok in enumerate(r_toks) if tok in p_toks)
        recall = r_match / r_weight_total if r_weight_total > 0 else 0.0

        # precision lato prediction
        p_weights = [token_weight(tok) for tok in p_toks]
        p_weight_total = sum(p_weights) or 1.0
        p_match = sum(p_weights[i] for i, tok in enumerate(p_toks) if tok in r_toks)
        precision = p_match / p_weight_total

        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)

    return float(np.mean(scores)) if scores else 0.0


# =========================
# METRICHE DI TEST
# =========================

def compute_metrics(preds, refs):
    """Calcola tutte le metriche richieste a partire da liste di stringhe."""
    assert len(preds) == len(refs)

    # corpus BLEU (nltk)
    preds_tokens = [p.split() for p in preds]
    refs_tokens = [[r.split()] for r in refs]
    bleu = corpus_bleu(refs_tokens, preds_tokens)

    # ROUGE
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        s = rouge.score(r, p)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(refs)
    r1, r2, rl = r1 / n, r2 / n, rl / n

    # METEOR
    meteor = np.mean(
        [meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]
    )

    # SacreBLEU (in scala 0–100 qui, come prima per RACE)
    sacre = sacrebleu.corpus_bleu(preds, [refs]).score

    # Identifier recall
    id_rec = identifier_recall(preds, refs)

    # CIDEr (se disponibile)
    cider = compute_cider(preds, refs)

    # CodeBLEU "testuale" (approssimato)
    codebleu_t = compute_codebleu_text(preds, refs)

    return dict(
        BLEU=float(bleu),
        METEOR=float(meteor),
        ROUGE1=float(r1),
        ROUGE2=float(r2),
        ROUGEL=float(rl),
        SACREBLEU=float(sacre),
        IDENTIFIER_RECALL=float(id_rec),
        CodeBLEU_text=float(codebleu_t),
        CIDEr=float(cider) if cider is not None else None,
    )


# =========================
# VALUTAZIONE PER UN SEED
# =========================

def evaluate_seed(seed: int):
    run_dir = RUN_DIR_TPL.format(seed=seed)
    out_dir = os.path.join(SAVED_MODEL_DIR, run_dir)

    gold_path = os.path.join(out_dir, "test.gold")
    pred_path = os.path.join(out_dir, "test.output")

    if not (os.path.exists(gold_path) and os.path.exists(pred_path)):
        print(f"[seed {seed}] ⚠️ test.gold/test.output non trovati in {out_dir}, salto.")
        return None

    refs = read_lines(gold_path)
    preds = read_lines(pred_path)

    if len(preds) != len(refs):
        m = min(len(preds), len(refs))
        print(
            f"[seed {seed}] Attenzione: len(preds)={len(preds)}, "
            f"len(refs)={len(refs)}. Tronco a {m}."
        )
        preds = preds[:m]
        refs = refs[:m]

    print(f"[seed {seed}] Test examples: {len(refs)}")

    metrics = compute_metrics(preds, refs)
    metrics["seed"] = seed
    return metrics, preds, refs


# =========================
# MAIN
# =========================

def main():
    all_metrics = []

    for seed in SEEDS:
        set_seed(seed)

        res = evaluate_seed(seed)
        if res is None:
            continue

        metrics, preds, refs = res
        all_metrics.append(metrics)

        base = os.path.join(RESULTS_DIR, f"race_50k_seed{seed}")
        # salvataggio metriche e predizioni
        with open(base + "_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(base + "_preds.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(preds))
        with open(base + "_refs.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(refs))

        print(f"[seed {seed}] Metriche:", metrics)

    if not all_metrics:
        print("❌ Nessuna metrica calcolata (nessun seed valido).")
        return

    # dataframe riepilogativo
    df = pd.DataFrame(all_metrics).set_index("seed")
    csv_path = os.path.join(RESULTS_DIR, "race_50k_metrics_all_seeds.csv")
    df.to_csv(csv_path)
    print("\n✅ CSV con tutte le metriche salvato in:", csv_path)

    # seed migliore in base al SacreBLEU
    best_seed = max(all_metrics, key=lambda m: m["SACREBLEU"])["seed"]
    best_metrics = [m for m in all_metrics if m["seed"] == best_seed][0]

    best_json = os.path.join(RESULTS_DIR, "race_50k_best_seed_metrics.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n🌟 Miglior seed (SacreBLEU): {best_seed}")
    print("   Metriche migliori:", best_metrics)
    print("   Salvate in:", best_json)


if __name__ == "__main__":
    main()
