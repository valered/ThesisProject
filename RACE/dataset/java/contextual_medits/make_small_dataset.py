import json
from pathlib import Path
import random

random.seed(42)  # così è ripetibile

BASE = Path(__file__).parent  # la cartella dove sta lo script

def make_small(src_name, dst_name, max_examples):
    src = BASE / src_name
    dst = BASE / dst_name
    print(f"Subsampling {src} -> {dst} (max {max_examples})")

    with src.open("r", encoding="utf-8") as fin, \
         dst.open("w", encoding="utf-8") as fout:
        lines = fin.readlines()
        if len(lines) > max_examples:
            lines = random.sample(lines, max_examples)
        for line in lines:
            obj = json.loads(line)
            json.dump(obj, fout)
            fout.write("\n")

make_small("train.jsonl", "train_small.jsonl", 50000)   # adattabile
make_small("valid.jsonl", "valid_small.jsonl",  5000)
make_small("test.jsonl",  "test_small.jsonl",   5000)