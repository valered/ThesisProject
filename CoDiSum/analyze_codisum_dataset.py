import os
import sys
from collections import Counter
import math

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]

def basic_stats(texts):
    n = len(texts)
    lengths = [len(t.split()) for t in texts]
    if n == 0:
        return dict(n=0, avg_len=0, min_len=0, max_len=0, uniq=0)
    return dict(
        n=n,
        avg_len=sum(lengths) / n,
        min_len=min(lengths),
        max_len=max(lengths),
        uniq=len(set(texts)),
    )

def show_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def analyze_split(base_dir, split_name):
    src_path = os.path.join(base_dir, f"{split_name}.txt.src")
    tgt_path = os.path.join(base_dir, f"{split_name}.txt.tgt")

    if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
        print(f"[{split_name}] File mancanti: {src_path} / {tgt_path}")
        return

    src = read_lines(src_path)
    tgt = read_lines(tgt_path)

    assert len(src) == len(tgt), f"{split_name}: src e tgt hanno lunghezze diverse!"

    show_header(f"SPLIT: {split_name}  (n = {len(src)})")

    # --- STATS DI BASE ---
    src_stats = basic_stats(src)
    tgt_stats = basic_stats(tgt)

    print("Diff (SRC):")
    print(f"  n esempi        : {src_stats['n']}")
    print(f"  lunghezza media : {src_stats['avg_len']:.2f} token")
    print(f"  min / max len   : {src_stats['min_len']} / {src_stats['max_len']}")
    print(f"  linee uniche    : {src_stats['uniq']} ({src_stats['uniq'] / src_stats['n'] * 100:.2f}%)")

    print("\nMessaggi (TGT):")
    print(f"  n esempi        : {tgt_stats['n']}")
    print(f"  lunghezza media : {tgt_stats['avg_len']:.2f} token")
    print(f"  min / max len   : {tgt_stats['min_len']} / {tgt_stats['max_len']}")
    print(f"  linee uniche    : {tgt_stats['uniq']} ({tgt_stats['uniq'] / tgt_stats['n'] * 100:.2f}%)")

    # --- TOKEN STATS ---
    src_tokens = []
    tgt_tokens = []
    for s in src:
        src_tokens.extend(s.split())
    for t in tgt:
        tgt_tokens.extend(t.split())

    src_vocab = Counter(src_tokens)
    tgt_vocab = Counter(tgt_tokens)

    print("\nVocab diff (SRC):")
    print(f"  # token totali : {len(src_tokens)}")
    print(f"  # tipi unici   : {len(src_vocab)}")
    print(f"  type/token ratio: {len(src_vocab)/len(src_tokens):.4f}")

    print("\nVocab messaggi (TGT):")
    print(f"  # token totali : {len(tgt_tokens)}")
    print(f"  # tipi unici   : {len(tgt_vocab)}")
    print(f"  type/token ratio: {len(tgt_vocab)/len(tgt_tokens):.4f}")

    # --- TOP MESSAGGI RIPETUTI ---
    tgt_counts = Counter(tgt)
    print("\nTop 20 messaggi più frequenti (per individuare pattern tipo 'fix bug'):")
    for msg, c in tgt_counts.most_common(20):
        print(f"  {c:5d}  | {msg}")

    # --- SAMPLE CASUALI ---
    print("\nEsempi casuali:")
    import random
    for _ in range(3):
        i = random.randrange(len(src))
        print(f"\n  [Esempio {i}]")
        print("  SRC:", src[i][:300])
        print("  TGT:", tgt[i])

def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_codisum_dataset.py <cartella_con_txt>")
        print("Esempio: python analyze_codisum_dataset.py processed")
        sys.exit(1)

    base_dir = sys.argv[1]

    for split in ["train", "valid", "test"]:
        analyze_split(base_dir, split)

if __name__ == "__main__":
    main()
