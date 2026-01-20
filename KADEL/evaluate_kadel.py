import os
import json
import math
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# =========  Tentativo di importare SacreBLEU  =========
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:  # pragma: no cover
    sacrebleu = None
    HAS_SACREBLEU = False
    print("⚠️ sacrebleu non installato: SACREBLEU verrà impostato a NaN.")

# =========  Tentativo di importare CIDEr (pycocoevalcap)  =========
try:
    from pycocoevalcap.cider.cider import Cider
    HAS_CIDER = True
except ImportError:  # pragma: no cover
    Cider = None
    HAS_CIDER = False
    print("⚠️ pycocoevalcap non installato: CIDEr verrà impostato a NaN.")


# =========  PATH  =========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results_kadel")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========  UTILS DI BASE  =========

def normalize_for_tokenization(s: str) -> str:
    """Pulizia minima: strip e normalizzazione spazi."""
    return " ".join(s.strip().split())


def tokenize(s: str) -> List[str]:
    """Tokenizzazione semplice per commit message."""
    return normalize_for_tokenization(s).split()


# =========  IDENTIFIER RECALL (stile KADEL)  =========

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_identifiers(text: str) -> List[str]:
    """Estrae token che sembrano identificatori (stile Java) dal testo."""
    return _IDENTIFIER_RE.findall(text)


def compute_identifier_recall(references: List[str], hypotheses: List[str]) -> float:
    assert len(references) == len(hypotheses)
    recalls = []
    for ref, hyp in zip(references, hypotheses):
        ref_ids = set(extract_identifiers(ref))
        hyp_ids = set(extract_identifiers(hyp))
        if not ref_ids:
            # se il riferimento non contiene identificatori, lo ignoriamo
            continue
        inter = len(ref_ids & hyp_ids)
        rec = inter / len(ref_ids)
        recalls.append(rec)
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


# =========  CIDEr  =========

def compute_cider(references: List[str], hypotheses: List[str]) -> Optional[float]:
    """CIDEr via pycocoevalcap; se la libreria manca, restituisce None."""
    if not HAS_CIDER:
        return None

    assert len(references) == len(hypotheses)
    gts, res = {}, {}
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        key = str(i)
        gts[key] = [ref]
        res[key] = [hyp]

    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return float(score)


# =========  CodeBLEU "testuale" (stessa definizione di RACE)  =========

def compute_codebleu_text(hypotheses: List[str], references: List[str]) -> float:
    """
    Versione 'testuale' in stile CoDiSum/RACE:
    pesi lessicali in base alla frequenza e F1 pesato.
    Non richiede la libreria ufficiale CodeBLEU.
    """
    # frequenza di tutti i token in predizioni + reference
    freq: Dict[str, int] = {}
    for txt in list(hypotheses) + list(references):
        for tok in txt.split():
            freq[tok] = freq.get(tok, 0) + 1

    def token_weight(tok: str) -> float:
        f = freq.get(tok, 1)
        return 1.0 / np.log2(f + 1.5)

    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_toks = hyp.split()
        ref_toks = ref.split()
        if not ref_toks:
            continue

        # lato reference (recall)
        w_r = {i: token_weight(tok) for i, tok in enumerate(ref_toks)}
        r_weight_total = sum(w_r.values())
        r_match = sum(
            w_r[i] for i, tok in enumerate(ref_toks) if tok in hyp_toks
        )
        recall = r_match / r_weight_total if r_weight_total > 0 else 0.0

        # lato prediction (precision)
        p_weights = [token_weight(tok) for tok in hyp_toks]
        p_weight_total = sum(p_weights) or 1.0
        p_match = sum(
            p_weights[i] for i, tok in enumerate(hyp_toks) if tok in ref_toks
        )
        precision = p_match / p_weight_total

        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)

    return float(np.mean(scores)) if scores else 0.0


# =========  METRICHE NLG  =========

def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    assert len(references) == len(hypotheses)

    # BLEU (NLTK) – stesso uso di RACE, senza smoothing
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]
    bleu = corpus_bleu(refs_tok, hyps_tok)

    # METEOR (NLTK vuole frasi già tokenizzate)
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
    meteor = float(np.mean(meteor_scores))

    # ROUGE-1 / 2 / L
    r1_scores, r2_scores, rl_scores = [], [], []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        r1_scores.append(s["rouge1"].fmeasure)
        r2_scores.append(s["rouge2"].fmeasure)
        rl_scores.append(s["rougeL"].fmeasure)

    rouge1 = float(np.mean(r1_scores))
    rouge2 = float(np.mean(r2_scores))
    rougeL = float(np.mean(rl_scores))

    # SacreBLEU (0–100, come in RACE)
    if HAS_SACREBLEU:
        sb = sacrebleu.corpus_bleu(hypotheses, [references])
        sacre_bleu = float(sb.score)
    else:
        sacre_bleu = float("nan")

    # Identifier recall
    id_recall = compute_identifier_recall(references, hypotheses)

    # CIDEr
    cider_val = compute_cider(references, hypotheses)

    # CodeBLEU testo
    codebleu_t = compute_codebleu_text(hypotheses, references)

    return {
        "BLEU": float(bleu),
        "METEOR": meteor,
        "ROUGE1": rouge1,
        "ROUGE2": rouge2,
        "ROUGEL": rougeL,
        "SACREBLEU": sacre_bleu,
        "IDENTIFIER_RECALL": id_recall,
        "CodeBLEU_text": codebleu_t,
        "CIDEr": float(cider_val) if cider_val is not None and not math.isnan(cider_val) else None,
    }


# =========  SCELTA FILE MIGLIORI DENTRO pred/  =========

def pick_eval_files(pred_dir: str) -> Tuple[str, str]:
    """
    Sceglie automaticamente quali file usare come gold/pred.

    Priorità:
    1) se esistono test.gold / test.output li usa direttamente;
    2) altrimenti cerca eval_ppl_result_clean_*.loss e prende l'epoch
       con loss minima, usando i corrispondenti .gold/.output;
    3) se non trova i 'clean', prova con eval_ppl_result_*.loss normali.
    """
    test_gold = os.path.join(pred_dir, "test.gold")
    test_out = os.path.join(pred_dir, "test.output")
    if os.path.exists(test_gold) and os.path.exists(test_out):
        return test_gold, test_out

    files = os.listdir(pred_dir)

    def _search_pattern(prefix: str) -> Optional[Tuple[str, str]]:
        """Ritorna (gold_path, out_path) per il migliore epoch secondo la loss."""
        pattern = re.compile(rf"{prefix}_(\d+)\.loss$")
        best_epoch = None
        best_loss = None

        for fname in files:
            m = pattern.match(fname)
            if not m:
                continue
            epoch = int(m.group(1))
            loss_path = os.path.join(pred_dir, fname)
            try:
                with open(loss_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    loss_val = float(content.split()[0])
            except Exception:
                continue

            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch

        if best_epoch is None:
            return None

        gold_path = os.path.join(pred_dir, f"{prefix}_{best_epoch}.gold")
        out_path = os.path.join(pred_dir, f"{prefix}_{best_epoch}.output")

        if not (os.path.exists(gold_path) and os.path.exists(out_path)):
            return None
        return gold_path, out_path

    # 1) prova con eval_ppl_result_clean_X.*
    res = _search_pattern("eval_ppl_result_clean")
    if res is not None:
        return res

    # 2) fallback: eval_ppl_result_X.*
    res = _search_pattern("eval_ppl_result")
    if res is not None:
        return res

    raise FileNotFoundError(
        f"Nessun file test.* né eval_ppl_result(_clean)_*.{{gold,output,loss}} trovato in {pred_dir}"
    )


def load_predictions_and_references(pred_dir: str) -> Tuple[List[str], List[str]]:
    """
    Usa pick_eval_files per determinare quali file gold/output usare,
    poi li carica e restituisce (references, hypotheses).
    """
    gold_path, pred_path = pick_eval_files(pred_dir)

    with open(gold_path, "r", encoding="utf-8") as fg:
        references = [l.rstrip("\n") for l in fg]

    with open(pred_path, "r", encoding="utf-8") as fp:
        hypotheses = [l.rstrip("\n") for l in fp]

    n = min(len(references), len(hypotheses))
    references = references[:n]
    hypotheses = hypotheses[:n]

    return references, hypotheses


# =========  EVALUATION PER SEED  =========

def evaluate_seed(seed: int) -> Optional[Tuple[Dict[str, float], List[str], List[str]]]:
    run_dir = os.path.join(SAVED_MODELS_DIR, f"KADEL_seed{seed}")
    pred_dir = os.path.join(run_dir, "pred")

    if not os.path.isdir(pred_dir):
        print(f"[seed {seed}] ⚠️ cartella pred non trovata in {run_dir}, salto.")
        return None

    try:
        references, hypotheses = load_predictions_and_references(pred_dir)
    except FileNotFoundError as e:
        print(f"[seed {seed}] ⚠️ {e}")
        return None

    print(f"[seed {seed}] Test examples: {len(references)}")

    metrics = compute_metrics(references, hypotheses)
    metrics["seed"] = seed
    print(f"[seed {seed}] METRICHE: {json.dumps(metrics, indent=2)}")
    return metrics, hypotheses, references


def main():
    seeds = [42, 123, 999]
    all_metrics: List[Dict[str, float]] = []

    for seed in seeds:
        res = evaluate_seed(seed)
        if res is None:
            continue

        metrics, preds, refs = res
        all_metrics.append(metrics)

        base = os.path.join(RESULTS_DIR, f"kadel_seed{seed}")
        # metriche
        with open(base + "_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        # predizioni e reference (come per RACE)
        with open(base + "_preds.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(preds))
        with open(base + "_refs.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(refs))

        print(f"[seed {seed}] Metriche:", metrics)

    if not all_metrics:
        print("❌ Nessuna metrica calcolata (nessun seed valido).")
        return

    # ---------- CSV con tutte le metriche ----------
    df = pd.DataFrame(all_metrics).set_index("seed")
    csv_path = os.path.join(RESULTS_DIR, "kadel_metrics_all_seeds.csv")
    df.to_csv(csv_path)
    print("\n✅ CSV con tutte le metriche salvato in:", csv_path)

    # ---------- Seed migliore (in base a SACREBLEU) ----------
    best_seed = max(all_metrics, key=lambda m: m["SACREBLEU"])["seed"]
    best_metrics = [m for m in all_metrics if m["seed"] == best_seed][0]
    best_json = os.path.join(RESULTS_DIR, "kadel_best_seed_metrics.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n🌟 Miglior seed (SACREBLEU): {best_seed}")
    print("   Metriche migliori:", best_metrics)
    print("   Salvate in:", best_json)

    # ---------- Summary mean/std ----------
    keys = [k for k in all_metrics[0].keys() if k != "seed"]
    summary = {}
    for k in keys:
        values = [m[k] for m in all_metrics]
        mean = float(np.mean(values))
        std = float(np.std(values))
        summary[k] = {"mean": mean, "std": std}

    print("\n===== RIEPILOGO FINALE KADEL (media ± std sulle run) =====")
    for k, v in summary.items():
        if v["std"] is None or (isinstance(v["std"], float) and math.isnan(v["std"])):
            print(f"{k}: {v['mean']}")
        else:
            print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

    with open(os.path.join(RESULTS_DIR, "kadel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
