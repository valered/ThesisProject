from keras import Model
from keras.layers import Input, Embedding, Bidirectional, GRU, Dense, TimeDistributed, Concatenate, Lambda, Dropout
import keras.backend as K
import numpy as np
from attention import Masked, MaskedTimeAttentionWithCoverage, MaskedGlobalMaxPooling1D, MaskedGlobalAveragePooling1D
from defined_layers import GetPiece, AttentionCopy, CombineGenCopy, MaskedSoftmax, ComputeAlpha, WeightedSum
from defined_layers import MaskedConv2D, MaskedAveragePooling2D, ComputeAttention, MaskedCopyProb
import tensorflow as tf
from tensorflow.keras import mixed_precision

# === Mixed precision (coerente con training su GPU/Colab) ===
mixed_precision.set_global_policy('mixed_float16')


def make_gru(units, return_sequences=True, return_state=True, bidirectional=False):
    gru = tf.keras.layers.GRU(
        units,
        return_sequences=return_sequences,
        return_state=return_state,
        recurrent_activation='sigmoid',
        reset_after=True
    )
    if bidirectional:
        return tf.keras.layers.Bidirectional(gru)
    return gru


def CopyNetPlus(len_en, len_de, attr_num, embed_vocab_size, decode_vocab_size,
                m_embed_dim, w_embed_dim, hid_size=192, att_num=64, drop_rate=0.1,
                gen_mask=None, copy_mask=None):
    """
    Versione aggiornata e coerente con il training:
    - hid_size = 192
    - dropout = 0.1
    - mixed precision abilitata
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Embedding, GRU, Dense, Dropout,
                                         Bidirectional, Concatenate, Lambda, TimeDistributed)
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
    import numpy as np

    # ---------- LAYER BASE ----------
    a = np.random.random([m_embed_dim]).astype("float32")
    b = np.zeros([m_embed_dim], dtype="float32")
    weight = np.stack([b, -a, b, a]).astype("float32")  # (4, m_embed_dim)

    mark_embed_layer = Embedding(
        input_dim=4,
        output_dim=m_embed_dim,
        mask_zero=False,
        weights=[weight],
        trainable=True,
        name="mark_embed"
    )
    word_embed_layer = Embedding(
        input_dim=embed_vocab_size,
        output_dim=w_embed_dim,
        mask_zero=False,
        name="word_embed"
    )

    # Encoder (bi-GRU)
    bi_rnn_layer1 = Bidirectional(GRU(hid_size, return_sequences=True, return_state=True,
                                      recurrent_dropout=drop_rate, reset_after=False), name="bi_gru_1")
    bi_rnn_layer2 = Bidirectional(GRU(hid_size, return_sequences=True, return_state=True,
                                      recurrent_dropout=drop_rate, reset_after=False), name="bi_gru_2")
    bi_rnn_layer3 = Bidirectional(GRU(hid_size, return_sequences=True, return_state=True,
                                      recurrent_dropout=drop_rate, reset_after=False), name="bi_gru_3")

    bi_rnn_layer4 = Bidirectional(GRU(hid_size, return_sequences=True,
                                      recurrent_dropout=drop_rate, reset_after=False), name="bi_gru_attr_1")
    bi_rnn_layer5 = Bidirectional(GRU(hid_size, return_sequences=True,
                                      recurrent_dropout=drop_rate, reset_after=False), name="bi_gru_attr_2")
    bi_rnn_layer6 = Bidirectional(GRU(hid_size, return_sequences=False,
                                      recurrent_dropout=drop_rate, reset_after=False), name="bi_gru_attr_3")

    # Decoder (uni-GRU)
    rnn_layer1 = GRU(hid_size * 2, return_sequences=True, return_state=True,
                     recurrent_dropout=drop_rate, reset_after=False, name="dec_gru_1")
    rnn_layer2 = GRU(hid_size * 2, return_sequences=True, return_state=True,
                     recurrent_dropout=drop_rate, reset_after=False, name="dec_gru_2")
    rnn_layer3 = GRU(hid_size * 2, return_sequences=True, return_state=True,
                     recurrent_dropout=drop_rate, reset_after=False, name="dec_gru_3")

    compute_alpha = ComputeAttention(att_num)
    p_gen_dense_layer = Dense(1, activation='sigmoid', name="pgen_dense")
    gen_dense_layer = Dense(decode_vocab_size, name="gen_vocab_dense")
    dropout = Dropout(drop_rate, name="drop")

    # ---------- INPUT ----------
    m_encoder_in = Input(shape=(len_en,), dtype='int32', name="m_in")
    w_encoder_in = Input(shape=(len_en,), dtype='int32', name="w_in")
    a_encoder_in = Input(shape=(len_en, attr_num), dtype='float32', name="a_in")

    # ---------- MASK ----------
    mask_2d = Lambda(lambda t: tf.cast(tf.not_equal(t, 0), tf.float32),
                     name="seq_mask")(w_encoder_in)
    mask_3d = Lambda(lambda m: tf.expand_dims(m, -1),
                     output_shape=lambda s: (s[0], s[1], 1),
                     name="mask_to_3d")(mask_2d)

    # ---------- ATTRIBUTI ----------
    a_embed_en = TimeDistributed(Dense(m_embed_dim), name="attr_proj")(a_encoder_in)

    # ---------- ENCODER ----------
    m_embed_en = mark_embed_layer(m_encoder_in)
    w_embed_en = word_embed_layer(w_encoder_in)
    embed_en = Concatenate(name="enc_concat_mark_word")([m_embed_en, w_embed_en])

    rnn_h1, state_f1, state_b1 = bi_rnn_layer1(embed_en)
    rnn_h2, state_f2, state_b2 = bi_rnn_layer2(dropout(rnn_h1))
    rnn_h3, state_f3, state_b3 = bi_rnn_layer3(dropout(rnn_h2))

    # Attributi paralleli
    a_rnn_h1 = bi_rnn_layer4(a_embed_en)
    a_rnn_h2 = bi_rnn_layer5(dropout(a_rnn_h1))
    a_rnn_h3 = bi_rnn_layer6(dropout(a_rnn_h2))
    a_rnn_h3 = Lambda(
        lambda x: tf.repeat(tf.expand_dims(x, axis=1), repeats=len_en, axis=1),
        output_shape=(len_en, hid_size * 2),
        name="attr_repeat_to_L"
    )(a_rnn_h3)

    # Stati iniziali decoder
    state1 = Concatenate(name="dec_init_1")([state_f1, state_b1])
    state2 = Concatenate(name="dec_init_2")([state_f2, state_b2])
    state3 = Concatenate(name="dec_init_3")([state_f3, state_b3])

    rnn_h3 = dropout(rnn_h3)

    masked_rnn_h3, _ = Masked(return_mask=True, name="masked_enc")(rnn_h3)
    enc_hid = Concatenate(name="enc_cat_text_attr")([masked_rnn_h3, a_rnn_h3])

    encoder = Model(
        inputs=[m_encoder_in, w_encoder_in, a_encoder_in],
        outputs=[enc_hid, mask_2d, m_embed_en, state1, state2, state3],
        name="encoder"
    )

    # ---------- DECODER ----------
    decoder_in = Input(shape=(len_de,), dtype='int32', name="dec_in")
    embed_de = word_embed_layer(decoder_in)

    rnn_h4, _ = rnn_layer1(embed_de, initial_state=state1)
    rnn_h5, _ = rnn_layer2(dropout(rnn_h4), initial_state=state2)
    rnn_h6, _ = rnn_layer3(dropout(rnn_h5), initial_state=state3)
    rnn_h6 = dropout(rnn_h6)

    rnn_h6_proj = TimeDistributed(Dense(hid_size * 4), name="dec_proj_to_4hid")(rnn_h6)

    compute_alpha.supports_masking = False
    alpha = compute_alpha([enc_hid, rnn_h6_proj, mask_3d])

    att_cont = Lambda(lambda x: tf.einsum('btl,blh->bth', x[0], x[1]),
                      name="att_context_hid")([alpha, enc_hid])
    att_mark = Lambda(lambda x: tf.einsum('btl,blh->bth', x[0], x[1]),
                      name="att_context_mark")([alpha, m_embed_en])

    att_cont = dropout(att_cont)
    att_mark = dropout(att_mark)

    p_gen_source = Concatenate(name="pgen_src")([rnn_h6, att_cont, embed_de])
    p_gen = p_gen_dense_layer(p_gen_source)

    att_out = Concatenate(name="att_out")([rnn_h6, att_cont, att_mark])
    gen_prob = TimeDistributed(gen_dense_layer, name="gen_logits")(att_out)
    gen_prob = MaskedSoftmax(gen_mask, name="gen_prob")(gen_prob)

    copy_prob = AttentionCopy(decode_vocab_size, name="copy_raw")([w_encoder_in, alpha])
    copy_prob = MaskedCopyProb(copy_mask, name="copy_prob")(copy_prob)

    output = CombineGenCopy(name="mix_gen_copy")([p_gen, gen_prob, copy_prob])

    model = Model(
        inputs=[m_encoder_in, w_encoder_in, a_encoder_in, decoder_in],
        outputs=output,
        name="CoDiSum_CopyNetPlus"
    )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, encoder, None
