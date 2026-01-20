import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
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

# ===========================================================
# PATHS
# ===========================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "processed")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results_codisum")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_SRC_LEN  = 200
MAX_TGT_LEN  = 32
VOCAB_SIZE   = 30000
SEEDS        = [42, 123, 999]


# ===========================================================
# UTILS
# ===========================================================
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]

def load_tokenizers():
    with open(os.path.join(MODEL_DIR, "tokenizer_src.json"), encoding="utf-8") as f:
        tok_src = tokenizer_from_json(f.read())
    with open(os.path.join(MODEL_DIR, "tokenizer_tgt.json"), encoding="utf-8") as f:
        tok_tgt = tokenizer_from_json(f.read())
    return tok_src, tok_tgt

def sequences_from_texts(tok, texts, maxlen):
    seq = tok.texts_to_sequences(texts)
    arr = []
    vmax = VOCAB_SIZE - 1
    for s in seq:
        arr.append([i if i < VOCAB_SIZE else vmax for i in s])
    return pad_sequences(arr, maxlen=maxlen, padding="post", truncating="post")


# ===========================================================
# GREEDY DECODING
# ===========================================================
def greedy_decode(model, tok_tgt, Xenc):
    index_word = {v: k for k, v in tok_tgt.word_index.items()}
    bos_id = tok_tgt.word_index.get("<s>")
    eos_id = tok_tgt.word_index.get("</s>")

    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer target privo di <s> o </s>.")

    preds = []
    n = Xenc.shape[0]

    for i in tqdm(range(n), desc="Decoding"):
        src_seq = Xenc[i:i+1]

        dec_seq = np.zeros((1, MAX_TGT_LEN), dtype="int32")
        dec_seq[0, 0] = bos_id
        generated = []

        for t in range(1, MAX_TGT_LEN):
            pred = model.predict([src_seq, dec_seq], verbose=0)
            next_id = int(np.argmax(pred[0, t - 1]))

            if next_id == 0 or next_id == eos_id:
                break

            generated.append(next_id)
            dec_seq[0, t] = next_id

        toks = [index_word.get(j, "<unk>") for j in generated]
        preds.append(" ".join(toks).strip())

    return preds


# ===========================================================
# METRICS
# ===========================================================
def compute_metrics(preds, refs):
    # BLEU
    preds_tok = [p.split() for p in preds]
    refs_tok  = [[r.split()] for r in refs]
    bleu = corpus_bleu(refs_tok, preds_tok)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(refs)
    r1 /= n
    r2 /= n
    rl /= n

    # METEOR – **ORA TOKENIZZATO**
    meteor_scores = [
        meteor_score([r.split()], p.split())
        for p, r in zip(preds, refs)
    ]
    meteor = float(np.mean(meteor_scores))

    # SACREBLEU (stringhe, come per RACE/KADEL)
    sacre = sacrebleu.corpus_bleu(preds, [refs]).score

    # CIDEr (se disponibile)
    if HAS_CIDER:
        gts = {i: [refs[i]] for i in range(len(refs))}
        res = {i: [preds[i]] for i in range(len(preds))}
        cider_score, _ = Cider().compute_score(gts, res)
    else:
        cider_score = None

    return dict(
        BLEU=float(bleu),
        METEOR=float(meteor),
        ROUGE1=float(r1),
        ROUGE2=float(r2),
        ROUGEL=float(rl),
        SACREBLEU=float(sacre),
        CIDEr=float(cider_score) if cider_score is not None else None,
    )


# ===========================================================
# MAIN
# ===========================================================
def main():
    test_src = read_lines(os.path.join(DATA_DIR, "test.txt.src"))
    test_tgt = read_lines(os.path.join(DATA_DIR, "test.txt.tgt"))
    assert len(test_src) == len(test_tgt), "Mismatch src/tgt nel test set"
    print(f"Test set: {len(test_src)} esempi")

    tok_src, tok_tgt = load_tokenizers()
    Xenc = sequences_from_texts(tok_src, test_src, MAX_SRC_LEN)

    all_metrics = []

    for seed in SEEDS:
        print(f"\n=== CoDiSum — seed {seed} ===")
        set_seed(seed)

        base = os.path.join(RESULTS_DIR, f"codisum_seed{seed}")
        preds_path = base + "_preds.txt"
        refs_path  = base + "_refs.txt"
        metrics_path = base + "_metrics.json"

        # Se abbiamo già preds/refs, non rifacciamo il decoding
        if os.path.exists(preds_path) and os.path.exists(refs_path):
            print("  Preds/refs già presenti, li ricarico.")
            preds = read_lines(preds_path)
            refs = read_lines(refs_path)
        else:
            model_path = os.path.join(MODEL_DIR, f"codisum_best_seed{seed}.h5")
            if not os.path.exists(model_path):
                print(f"  ⚠️ Modello non trovato: {model_path}, salto.")
                continue

            model = load_model(model_path, compile=False)
            preds = greedy_decode(model, tok_tgt, Xenc)

            # 👉 Salviamo SUBITO preds/refs, così non si perde nulla
            with open(preds_path, "w", encoding="utf-8") as f:
                f.write("\n".join(preds))
            with open(refs_path, "w", encoding="utf-8") as f:
                f.write("\n".join(test_tgt))
            refs = test_tgt

        print(f"  Test examples: {len(preds)}")

        metrics = compute_metrics(preds, refs)
        metrics["seed"] = seed
        all_metrics.append(metrics)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print("  Metriche:", metrics)

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(RESULTS_DIR, "codisum_metrics_all_seeds.csv"), index=False)
        print("\n✅ Valutazione completata. Risultati salvati in:", RESULTS_DIR)
    else:
        print("❌ Nessuna metrica calcolata.")


if __name__ == "__main__":
    main()
