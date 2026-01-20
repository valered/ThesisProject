import numpy as np
from models1 import CopyNetPlus
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# === Stessi iperparametri del training ===
E_L = 150
D_L = 16
A_N = 5
WED = 150
HS = 100
MED = 50
ATN = 64
TR_DR = 0.1
VOCAB_TRAIN = 15000

# === Ricrea il modello ===
genmask = np.ones((VOCAB_TRAIN,), dtype="float32")
copymask = np.ones((VOCAB_TRAIN,), dtype="float32")

model, _, _ = CopyNetPlus(
    E_L, D_L, A_N,
    VOCAB_TRAIN, VOCAB_TRAIN,
    MED, WED, HS, ATN, TR_DR,
    genmask, copymask
)

# === Carica i pesi di una specifica epoca ===
checkpoint_path = "models/checkpoint_epoch_10_val1.650.weights.h5"
model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)


model.save_weights("CoDiSum_final_weights.h5")

print("✅ Modello ricostruito e pesi caricati!")
