import tensorflow as tf


def generate_padding_mask(input):
    result = tf.cast(input == 0, dtype=tf.float32)[:, tf.newaxis, tf.newaxis: ]
    return result

def generate_look_ahead_mask(input_length):
    mask = 1 - tf.linalg.band_part(tf.ones((input_length, input_length)), -1, 0)
    return mask

def generate_mask(inp, targ):
    encoder_padding_mask = generate_padding_mask(inp)
  
    # Decoder Padding Mask: Use for global multi head attention for masking encoder output
    decoder_padding_mask = generate_padding_mask(inp)
  
    # Look Ahead Padding Mask
    decoder_look_ahead_mask = generate_look_ahead_mask(targ.shape[1])
  
    # Decoder Padding Mask
    decoder_inp_padding_mask = generate_padding_mask(targ)

    decoder_look_ahead_mask = tf.maximum(decoder_look_ahead_mask, decoder_inp_padding_mask)

    return encoder_padding_mask, decoder_look_ahead_mask , decoder_padding_mask