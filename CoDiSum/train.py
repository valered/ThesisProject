# ==============================
# train.py — CoDiSum (seq2seq GRU + dot-attention)
# G02-compliant, Colab/Keras3 safe — ONLY TRAIN
# ==============================

import os, sys, json, random, time, socket, platform
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, Model
import argparse

# Evita incompatibilità Keras 3 con modelli seq2seq "classici"
tf.config.run_functions_eagerly(True)

# ------------------------------
# ARGOMENTI: SEED
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Seed per la run (es. 42, 123, 999)")
args = parser.parse_args()

# ------------------------------
# CONFIGURAZIONE (G02)
# ------------------------------
ENC_UNITS   = 192            # encoder hidden dim
DEC_UNITS   = 97             # decoder hidden dim (come G02)
EMB_DIM     = 192            # = ENC_UNITS per attenzione dot-product coerente
DROPOUT     = 0.10
LR          = 1e-3
BATCH_SIZE  = 100
EPOCHS      = 20
MAX_SRC_LEN = 200
MAX_TGT_LEN = 32
VOCAB_SIZE  = 30000
LOSS_FN     = "sparse_categorical_crossentropy"

BASE_DIR     = "/content/CoDiSum"
DATA_DIR     = os.path.join(BASE_DIR, "processed")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
RUN_ID       = time.strftime(f"run_%y%m%d_%H%M%S_seed{args.seed}")
RUN_DIR      = os.path.join(RESULTS_DIR, RUN_ID)

# Percorsi output
TRAIN_LOG_FILE   = os.path.join(RUN_DIR, "train_log.csv")
BEST_MODEL_FILE  = os.path.join(MODEL_DIR, f"codisum_best_seed{args.seed}.h5")
FINAL_MODEL_FILE = os.path.join(MODEL_DIR, f"codisum_final_seed{args.seed}.h5")
TOK_SRC_FILE     = os.path.join(MODEL_DIR, "tokenizer_src.json")
TOK_TGT_FILE     = os.path.join(MODEL_DIR, "tokenizer_tgt.json")
CFG_FILE         = os.path.join(RUN_DIR, "config_and_env.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

# ------------------------------
# UTILS
# ------------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]

def load_split(split):
    src = read_lines(os.path.join(DATA_DIR, f"{split}.txt.src"))
    tgt = read_lines(os.path.join(DATA_DIR, f"{split}.txt.tgt"))
    assert len(src) == len(tgt), f"Mismatch src/tgt in split {split}"
    return src, tgt

def build_tokenizer():
    # niente mask a livello di embedding; qui normale tokenizer
    return tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>", filters="")

def fit_tokenizer(tok, texts): tok.fit_on_texts(texts)

def tokenizer_to_json(tok, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(tok.to_json())

SPECIAL_TOKENS = {"pad":"<pad>", "bos":"<s>", "eos":"</s>", "unk":"<unk>"}

def sequences_from_texts(tok, texts, maxlen):
    seq = tok.texts_to_sequences(texts)
    arr = []
    vmax = VOCAB_SIZE - 1
    for s in seq:
        arr.append([i if i < VOCAB_SIZE else vmax for i in s])
    return tf.keras.preprocessing.sequence.pad_sequences(arr, maxlen=maxlen, padding="post", truncating="post")


def prepare_decoder_sequences(tok_tgt, tgt_texts, maxlen):
    bos, eos = SPECIAL_TOKENS["bos"], SPECIAL_TOKENS["eos"]
    dec_in  = [f"{bos} {t}" for t in tgt_texts]
    dec_out = [f"{t} {eos}" for t in tgt_texts]
    X = sequences_from_texts(tok_tgt, dec_in,  maxlen)
    y = sequences_from_texts(tok_tgt, dec_out, maxlen)
    return X, y

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------------------
# MODELLO — no mask_zero, init decoder via pooling (rank-2 garantito)
# ------------------------------
def build_codisum(vocab_in, vocab_out):
    # Encoder
    enc_in  = layers.Input(shape=(None,), name="enc_in")
    enc_emb = layers.Embedding(vocab_in, EMB_DIM, mask_zero=False, name="enc_emb")(enc_in)

    bi_gru = layers.Bidirectional(
        layers.GRU(
            ENC_UNITS,
            return_sequences=True,
            return_state=False,   # non usiamo più gli stati
            dropout=DROPOUT,
            name="encoder_gru"
        ),
        merge_mode="sum",
        name="bi_enc"
    )
    enc_out = bi_gru(enc_emb)  # (B, Tsrc, ENC_UNITS)

    # Inizializzazione decoder dallo "summary" dell'encoder (rank-2 fisso)
    enc_summary = layers.GlobalAveragePooling1D(name="enc_summary")(enc_out)  # (B, ENC_UNITS)
    init_state  = layers.Dense(DEC_UNITS, activation="tanh", name="state_proj")(enc_summary)

    # Decoder
    dec_in  = layers.Input(shape=(None,), name="dec_in")
    dec_emb = layers.Embedding(vocab_out, EMB_DIM, mask_zero=False, name="dec_emb")(dec_in)  # (B, Tdec, EMB_DIM)

    # Dot-attention (Luong-style) robusta (no mask)
    attn_scores  = layers.Dot(axes=[2, 2], name="attn_scores")([dec_emb, enc_out])   # (B, Tdec, Tsrc)
    attn_weights = layers.Activation("softmax", name="attn_weights")(attn_scores)
    attn_ctx     = layers.Dot(axes=[2, 1], name="attn_context")([attn_weights, enc_out])  # (B, Tdec, ENC_UNITS)

    dec_concat = layers.Concatenate(name="concat_ctx")([dec_emb, attn_ctx])  # (B, Tdec, EMB_DIM+ENC_UNITS)

    dec_out = layers.GRU(
        DEC_UNITS,
        return_sequences=True,
        dropout=DROPOUT,
        name="decoder_gru"
    )(dec_concat, initial_state=init_state)

    logits = layers.TimeDistributed(
        layers.Dense(vocab_out, activation="softmax"), name="out_dense"
    )(dec_out)

    model = Model([enc_in, dec_in], logits, name="CoDiSum")
    opt   = optimizers.RMSprop(learning_rate=LR, rho=0.9, clipnorm=1.0)
    model.compile(optimizer=opt, loss=LOSS_FN)
    return model

# ------------------------------
# MAIN
# ------------------------------
def main():
    set_seed(args.seed)

    device = tf.config.list_physical_devices("GPU")
    env_info = {
        "seed": args.seed,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "tf_version": tf.__version__,
        "gpu_present": bool(device),
        "gpus": [str(d) for d in device],
        "config": dict(
            ENC_UNITS=ENC_UNITS, DEC_UNITS=DEC_UNITS, EMB_DIM=EMB_DIM,
            DROPOUT=DROPOUT, LR=LR, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS,
            MAX_SRC_LEN=MAX_SRC_LEN, MAX_TGT_LEN=MAX_TGT_LEN, VOCAB_SIZE=VOCAB_SIZE
        ),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    save_json(env_info, CFG_FILE)

    # Dati
    src_train, tgt_train = load_split("train")
    src_val,   tgt_val   = load_split("valid")

    # Tokenizer
    tok_src, tok_tgt = build_tokenizer(), build_tokenizer()
    specials = " ".join(SPECIAL_TOKENS.values())
    fit_tokenizer(tok_src, src_train + src_val + [specials])
    fit_tokenizer(tok_tgt, [f"{SPECIAL_TOKENS['bos']} {t} {SPECIAL_TOKENS['eos']}" for t in tgt_train + tgt_val] + [specials])
    tokenizer_to_json(tok_src, TOK_SRC_FILE)
    tokenizer_to_json(tok_tgt, TOK_TGT_FILE)

    # Sequenze
    Xenc_tr = sequences_from_texts(tok_src, src_train, MAX_SRC_LEN)
    Xenc_va = sequences_from_texts(tok_src, src_val,   MAX_SRC_LEN)
    Xdec_tr, y_tr = prepare_decoder_sequences(tok_tgt, tgt_train, MAX_TGT_LEN)
    Xdec_va, y_va = prepare_decoder_sequences(tok_tgt, tgt_val,   MAX_TGT_LEN)

    # Modello + training
    model = build_codisum(VOCAB_SIZE, VOCAB_SIZE)
    cbs = [
        callbacks.ModelCheckpoint(BEST_MODEL_FILE, save_best_only=True, monitor="val_loss"),
        callbacks.CSVLogger(TRAIN_LOG_FILE)
    ]

    print(f"\n=== TRAINING (seed {args.seed}) ===")
    model.fit(
        [Xenc_tr, Xdec_tr], y_tr,
        validation_data=([Xenc_va, Xdec_va], y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    print("\n✅ Training completato. Salvataggio modello finale...")
    model.save(FINAL_MODEL_FILE)
    print(f"Output salvato in: {RUN_DIR}")

if __name__ == "__main__":
    main()
