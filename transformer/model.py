from keras.models import Model
from transformer.layers.encoder import Encoder
from transformer.layers.decoder import Decoder
from transformer.behaviors.position_wise_feed_forward_networks import ffn
from transformer.behaviors.mask import generate_mask
from keras.metrics import Mean
from keras.optimizers import Adam
import tensorflow as tf

class Transformer(Model):
    def __init__(self, n, h, input_vocab_size, target_vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(h, d_ff, d_model, input_vocab_size, activation, n, dropout_rate, eps)
        self.decoder = Decoder(h, d_ff, d_model, target_vocab_size, activation, n, dropout_rate, eps)
        self.ffn = ffn(d_ff, d_model, activation)
        self.optimizer = Adam(learning_rate=0.01)

        self.train_accuracy = Mean(name='train_accuracy')
        self.loss_accuracy = Mean(name='train_loss')
        
        self.checkpoint = tf.train.Checkpoint(model=self, optimizer = self.optimizer) 

    def call(self, encoder_in, decoder_in, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(encoder_in, is_train, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(decoder_in, encoder_output, decoder_look_ahead_mask, decoder_padding_mask)

        return self.ffn(decoder_output)