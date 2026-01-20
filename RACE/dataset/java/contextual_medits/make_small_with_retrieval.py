import json
from pathlib import Path
import random

random.seed(42)  # così il subset è ripetibile

BASE = Path(__file__).parent
RET_DIR = BASE / "codet5_retrieval_result"

def make_small_pair(split, max_examples):
    """
    split: 'train', 'valid', 'test'
    max_examples: numero massimo di esempi da tenere
    """
    src_main = BASE / f"{split}.jsonl"
    src_ret  = RET_DIR / f"{split}.jsonl"

    dst_main = BASE / f"{split}_small.jsonl"
    dst_ret  = RET_DIR / f"{split}_small.jsonl"

    print(f"\n=== {split.upper()} ===")
    print(f"Main: {src_main}")
    print(f"Retrieval: {src_ret}")

    # leggo tutte le righe
    with src_main.open("r", encoding="utf-8") as f_main, \
         src_ret.open("r", encoding="utf-8") as f_ret:
        main_lines = f_main.readlines()
        ret_lines  = f_ret.readlines()

    n_main = len(main_lines)
    n_ret  = len(ret_lines)
    print(f"Main lines: {n_main}, Retrieval lines: {n_ret}")

    if n_main != n_ret:
        raise ValueError(f"Lunghezze diverse per {split}: main={n_main}, retrieval={n_ret}. "
                         "RACE si aspetta che siano uguali.")

    k = min(max_examples, n_main)
    print(f"Subsampling {k} esempi (su {n_main})")

    indices = list(range(n_main))
    random.shuffle(indices)
    keep = set(indices[:k])

    with dst_main.open("w", encoding="utf-8") as f_main_out, \
         dst_ret.open("w", encoding="utf-8") as f_ret_out:
        for i in range(n_main):
            if i in keep:
                f_main_out.write(main_lines[i].rstrip("\n") + "\n")
                f_ret_out.write(ret_lines[i].rstrip("\n") + "\n")

    print(f"Creati:\n - {dst_main}\n - {dst_ret}")

if __name__ == "__main__":
    make_small_pair("train", 50000)
    make_small_pair("valid", 5000)
    make_small_pair("test",  5000)
