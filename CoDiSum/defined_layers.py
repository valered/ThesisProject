# defined_layers.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Input, Embedding, GRU, Dense,
    Conv2D, AveragePooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda
)
from tensorflow.keras import backend as K
import numpy as np


# ---------------------------
# Utility
# ---------------------------

def _to_same_dtype(t, ref):
    """Cast tensor t to the same dtype of ref tensor."""
    return tf.cast(t, ref.dtype)


# ---------------------------
# Layers
# ---------------------------

class GetPiece(Layer):
    def __init__(self, num, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.num = num

    def call(self, x, mask=None):
        return x[:, self.num, :, :]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[:, self.num, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])


class AttentionCopy(Layer):
    """
    inputs:
      - x[0]: encoder word ids (B, L) int
      - x[1]: attention weights alpha (B, T, L) float (fp16/fp32)
    output:
      - (B, T, |V|)
    """
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.supports_masking = False

    def call(self, inputs, **kwargs):
        enc_ids, alpha = inputs  # (B, L), (B, T, L)
        # one-hot nello stesso dtype di alpha
        one_hot = tf.one_hot(tf.cast(enc_ids, tf.int32), self.size, dtype=alpha.dtype)  # (B, L, |V|)
        # (B, T, L) @ (B, L, |V|) = (B, T, |V|)
        return tf.linalg.matmul(alpha, one_hot)

    def compute_output_shape(self, input_shape):
        # input_shape[1] = (B, T, L)
        return (input_shape[1][0], input_shape[1][1], self.size)


class ComputeAttention(Layer):
    """
    x[0]: en_seq  (B, L, H)
    x[1]: de_seq  (B, T, H)
    x[2]: mask    (B, L)   or (B, L, 1)  float {0.,1.}  (opzionale)
    output: alpha (B, T, L)
    """
    def __init__(self, att_num, **kwargs):
        super().__init__(**kwargs)
        self.att_num = att_num
        self.supports_masking = False

    def call(self, x, mask=None):
        en_seq = x[0]                               # (B, L, H)
        de_seq = x[1]                               # (B, T, H)
        m = None
        if len(x) >= 3 and x[2] is not None:
            m = x[2]                                # (B, L) o (B, L, 1)
        elif mask is not None:
            if isinstance(mask, (list, tuple)):
                mask = mask[0]
            m = tf.cast(mask, tf.float32)

        # att_scores: (B, T, L)
        att_scores = tf.linalg.matmul(de_seq, en_seq, transpose_b=True)

        if m is not None:
            # normalizza in (B, 1, L)
            if m.shape.rank == 2:
                m = tf.expand_dims(m, axis=1)       # (B, 1, L)
            elif m.shape.rank == 3 and m.shape[-1] == 1:
                m = tf.transpose(m, perm=[0, 2, 1]) # (B, 1, L)

            m = _to_same_dtype(m, att_scores)
            # grande penalità dove m==0
            big_neg = tf.cast(-1e6, att_scores.dtype)
            att_scores = att_scores + (1.0 - m) * big_neg

        return tf.nn.softmax(att_scores, axis=-1)

    def compute_output_shape(self, input_shape):
        # (B, T, L)
        return (input_shape[1][0], input_shape[1][1], input_shape[0][1])


class CombineGenCopy(Layer):
    """
    p*gen + (1-p)*copy
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        p_gen, gen_prob, copy_prob = inputs
        p_gen   = _to_same_dtype(p_gen, gen_prob)
        copy_prob = _to_same_dtype(copy_prob, gen_prob)
        return p_gen * gen_prob + (1.0 - p_gen) * copy_prob

    def compute_mask(self, inputs, mask=None):
        # Propaga eventualmente la mask della distribuzione generata
        if mask is None:
            return None
        if isinstance(mask, (list, tuple)) and len(mask) > 1:
            return mask[1]
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class Masked(Layer):
    """
    Applica una mask sui timestep non-zero e (se return_mask=True)
    restituisce anche la mask 2D in float (B, L).
    Evita problemi di Broadcast con mask booleane.
    """
    def __init__(self, return_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.return_mask = return_mask
        self.supports_masking = False

    def call(self, inputs, mask=None):
        x = tf.convert_to_tensor(inputs)
        x_dtype = x.dtype
        # timestep "attivo" se esiste almeno un feature != 0
        nonzero = tf.reduce_any(tf.not_equal(x, tf.zeros_like(x)), axis=-1)  # (B, L) bool
        mask_float = tf.cast(nonzero, x_dtype)                                # (B, L)
        y = x * tf.expand_dims(mask_float, axis=-1)                           # (B, L, F)
        if self.return_mask:
            # mask 2D in float (B, L) – più robusta
            return [y, mask_float]
        return y

    def compute_output_shape(self, input_shape):
        if self.return_mask:
            return [input_shape, (input_shape[0], input_shape[1])]
        return input_shape


class MaskedSoftmax(Layer):
    """
    Softmax sul vocabolario con mask (0/1) sul LAST AXIS.
    `mask` può essere numpy o tensor; viene sempre convertita al dtype degli inputs.
    """
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._mask_raw = tf.constant(mask)

    def call(self, inputs, **kwargs):
        # allinea la mask al dtype/logits
        m = tf.cast(self._mask_raw, inputs.dtype)               # (..., V) o (V,)
        if m.shape.rank == 1:
            # (V,) -> (1,1,V) per broadcast sicuro
            m = tf.reshape(m, (1, 1, -1))
        elif m.shape.rank == 2:
            m = tf.expand_dims(m, axis=0)                       # (1, V, ?) -> (non usato di solito)
        # grande penalità dove m==0
        big_neg = tf.cast(-1e6, inputs.dtype)
        logits = inputs + (1.0 - m) * big_neg
        return tf.nn.softmax(logits, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask": getattr(self, "mask", None),
            "size": getattr(self, "size", None),
            "att_num": getattr(self, "att_num", None),
        })
        return config



class MaskedCopyProb(Layer):
    """
    Applica mask sul VOCAB alle probabilità di copy.
    """
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._mask_raw = tf.constant(mask)

    def call(self, inputs, **kwargs):
        # inputs: (B, T, V)
        m = tf.cast(self._mask_raw, inputs.dtype)  # (V,) or (..,V)
        if m.shape.rank == 1:
            m = tf.reshape(m, (1, 1, -1))         # (1,1,V)
        elif m.shape.rank == 2:
            m = tf.expand_dims(m, axis=0)         # (1,*,V)
        return inputs * m

    def compute_output_shape(self, input_shape):
        return input_shape


class ComputeAlpha(Layer):
    """
    Legacy: prende N sequenze (B, L, H), le proietta con W -> (B, L, 1),
    concatena lungo una nuova dim e applica una mask (B, L) float.
    Output: softmax lungo L -> (B, L, N)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.W = self.add_weight(
            name='W',
            shape=(input_dim, 1),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # ultimo è la mask (B, L) o (B, L, 1)
        seqs = inputs[:-1]
        mask = inputs[-1]

        # proiezioni
        proj = []
        for s in seqs:
            # (B, L, H) @ (H, 1) -> (B, L, 1)
            p = tf.linalg.matmul(s, self.W)
            proj.append(p)

        # concat su ultima dim -> (B, L, N)
        outs = tf.concat(proj, axis=-1)

        # normalizza mask in (B, 1, L)
        if mask is not None:
            if mask.shape.rank == 3 and mask.shape[-1] == 1:
                mask = tf.squeeze(mask, axis=-1)  # (B, L)
            mask = tf.cast(mask, outs.dtype)
            mask = tf.expand_dims(mask, axis=1)   # (B, 1, L)

            big_neg = tf.cast(-1e6, outs.dtype)
            outs = tf.transpose(outs, [0, 2, 1])                # (B, N, L)
            outs = outs + (1.0 - mask) * big_neg
            outs = tf.transpose(outs, [0, 2, 1])                # (B, L, N)

        return tf.nn.softmax(outs, axis=1)  # softmax lungo L

    def compute_output_shape(self, input_shape):
        # (B, L, N)
        return (input_shape[0][0], input_shape[0][1], len(input_shape) - 1)


class WeightedSum(Layer):
    """
    inputs: [weights (B, K), x1 (B,D1), x2 (B,D2), ...]
    output: sum_k weights[:,k] * xk
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        weight = tf.expand_dims(inputs[0], axis=-1)  # (B, K, 1)
        xs = [tf.expand_dims(x, axis=1) for x in inputs[1:]]  # [(B,1,Di)...]
        x = tf.concat(xs, axis=1)                  # (B, K, D)
        weight = _to_same_dtype(weight, x)
        return tf.reduce_sum(x * weight, axis=1)   # (B, D)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class MaskedConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        super().__init__(*args, **kwargs)

    def call(self, x, mask=None):
        return super().call(x)

    def compute_mask(self, inputs, mask=None):
        return None


class MaskedAveragePooling2D(AveragePooling2D):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        super().__init__(*args, **kwargs)

    def call(self, x, mask=None):
        return super().call(x)

    def compute_mask(self, inputs, mask=None):
        return None


# ---------------------------------
# (opzionale) quick self-test
# ---------------------------------
def validate():
    B, L, T, H, V = 2, 5, 3, 4, 7
    # AttentionCopy test
    ids = tf.constant([[0,1,2,3,4],[1,2,3,4,5]], dtype=tf.int32)          # (B,L)
    alpha = tf.random.uniform((B, T, L), dtype=tf.float16)                 # (B,T,L) fp16
    ac = AttentionCopy(V)
    out = ac([ids, alpha])                                                 # (B,T,V)
    assert out.dtype == alpha.dtype

    # MaskedSoftmax test
    logits = tf.random.normal((B, T, V), dtype=tf.float16)
    m = np.ones((V,), dtype=np.float32)
    m[0] = 0
    ms = MaskedSoftmax(m)
    sm = ms(logits)
    assert sm.dtype == logits.dtype

    # ComputeAttention test
    en = tf.random.normal((B, L, H), dtype=tf.float32)
    de = tf.random.normal((B, T, H), dtype=tf.float32)
    mask = tf.ones((B, L), dtype=tf.float32)
    ca = ComputeAttention(att_num=1)
    a = ca([en, de, mask])
    assert a.shape == (B, T, L)

    print("validate(): OK")


if __name__ == "__main__":
    validate()
