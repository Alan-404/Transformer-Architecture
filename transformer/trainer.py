from transformer.behaviors.mask import generate_mask
import tensorflow as tf
from keras.metrics import Mean
from keras.losses import SparseCategoricalCrossentropy
from transformer.optimizer import CustomLearningRate
from keras.optimizers import Adam

class Trainer:
    def __init__(self, model, checkpoint_folder, epochs=10):
        self.model = model
        
        self.epochs = epochs
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = Mean(name='train_accuracy')
        self.lrate = CustomLearningRate(d_model=512)
        self.optimizer = Adam(learning_rate=self.lrate)
        self.checkpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_folder, max_to_keep=3)
        

    def cal_acc(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(real == 0)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.math.reduce_sum(accuracies) / tf.math.reduce_sum(mask)

    def loss_function(self, real, pred):
        
        cross_entropy = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        loss = cross_entropy(real, pred)
        
        mask = tf.cast(mask, dtype=loss.dtype)
        
        loss = loss*mask
        
        return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)


    def train_step(self, inp, targ):
        
        encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(inp, targ)

        with tf.GradientTape() as tape:
            preds = self.model(inp, targ, True, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)

            d_loss = self.loss_function(targ, preds)

        grads = tape.gradient(d_loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss.update_state(d_loss)
        self.train_accuracy.update_state(self.cal_acc(targ, preds))

    
    def fit(self, data):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        
        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, targ)) in enumerate(data):
                self.train_step(inp, targ)

                if batch %50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.3f} Accuracy {self.train_accuracy.result():.3f}')
                if (epoch + 1) % 5 == 0:
                    saved_path = self.checkpoint_manager.save()
                    print('Checkpoint was saved at {}'.format(saved_path))


    def predict(self, encoder_input, decoder_input, is_train, max_length, end_token):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        for i in range(max_length):
            encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(encoder_input, decoder_input)

            preds = self.model(encoder_input, decoder_input, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)

            preds = preds[:, -1:, :]

            predicted_id = tf.argmax(preds, axis=-1)
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

            if predicted_id == end_token:
                break
        return decoder_input