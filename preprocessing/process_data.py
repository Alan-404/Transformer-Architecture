import io
import tensorflow as tf
import re
import os
from sklearn.model_selection import train_test_split

import pickle

class Data:
  def __init__(self, inp_lang, targ_lang, vocab_folder):
    self.inp_lang = inp_lang
    self.targ_lang = targ_lang
    self.vocab_folder = vocab_folder
    self.inp_tokenizer_path = '{}{}_tokenizer.pickle'.format(self.vocab_folder, self.inp_lang)
    self.targ_tokenizer_path = '{}{}_tokenizer.pickle'.format(self.vocab_folder, self.targ_lang)
    
    self.inp_tokenizer = None
    self.targ_tokenizer = None

    if os.path.isfile(self.inp_tokenizer_path):
      # Loading tokenizer
      with open(self.inp_tokenizer_path, 'rb') as handle:
        self.inp_tokenizer = pickle.load(handle)

    if os.path.isfile(self.targ_tokenizer_path):
      # Loading tokenizer
      with open(self.targ_tokenizer_path, 'rb') as handle:
        self.targ_tokenizer = pickle.load(handle)


  def preprocess_sentence(self, w, max_length):
    w = str(w).lower().strip()
    w = re.sub(r"([?.!,¿])", "", w)
    w = re.sub(r'\s\s+', ' ', w.strip())

    # Truncate Length up to ideal_length
    w = " ".join(w.split()[:max_length+1])
    # Add start and end token 
    w = '{} {} {}'.format("<BOS>", w, "<EOS>")
    return w

  def build_tokenizer(self, lang_tokenizer, lang):

    if not lang_tokenizer:
      lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    lang_tokenizer.fit_on_texts(lang)
    return lang_tokenizer

  def tokenize(self, lang_tokenizer, lang, max_length):

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=max_length)
    return tensor


  def load_dataset(self, inp_path, targ_path, max_length, num_data = 10000):

    inp_lines = io.open(inp_path, encoding="UTF-8").read().strip().split('\n')[:num_data]
    targ_lines = io.open(targ_path, encoding="UTF-8").read().strip().split('\n')[:num_data]
    
    inp_lines = [self.preprocess_sentence(inp, max_length) for inp in inp_lines]
    targ_lines = [self.preprocess_sentence(targ, max_length) for targ in targ_lines]

    
    # Tokenizing
    self.inp_tokenizer = self.build_tokenizer(self.inp_tokenizer, inp_lines)
    inp_tensor = self.tokenize(self.inp_tokenizer, inp_lines, max_length)

    self.targ_tokenizer = self.build_tokenizer(self.targ_tokenizer, targ_lines)
    targ_tensor = self.tokenize(self.targ_tokenizer, targ_lines, max_length)

    # Saving Tokenizer
    print('=============Saving Tokenizer================')
    print('Begin...')

    if not os.path.exists(self.vocab_folder):
      try:
        os.makedirs(self.vocab_folder)
      except OSError as e: 
        raise IOError("Failed to create folders")

    with open(self.inp_tokenizer_path, 'wb') as handle:
      pickle.dump(self.inp_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(self.targ_tokenizer_path, 'wb') as handle:
      pickle.dump(self.targ_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done!!!')

    return inp_tensor, targ_tensor

  def build_dataset(self, inp_path, targ_path, buffer_size, batch_size, max_length, num_data=10000):

    inp_tensor, targ_tensor = self.load_dataset(inp_path, targ_path, max_length, num_data)

    inp_tensor_train, inp_tensor_val, targ_tensor_train, targ_tensor_val = train_test_split(inp_tensor, targ_tensor, test_size=0.2)

    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_tensor_train, dtype=tf.int64), tf.convert_to_tensor(targ_tensor_train, dtype=tf.int64)))

    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_tensor_val, dtype=tf.int64), tf.convert_to_tensor(targ_tensor_val, dtype=tf.int64)))

    val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)

    return train_dataset, val_dataset
