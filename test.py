#%%
from preprocessing.process_data import Data
from transformer.model import TransformerModel
from transformer.trainer import Trainer
# %%
data_processer = Data('en', 'vi', './check')
# %%
train, val = data_processer.build_dataset('./datasets/en_sents.txt', './datasets/vi_sents.txt', buffer_size=64, batch_size=64, max_length=40)
# %%
for batch, (inp, targ) in enumerate(train):
    print(inp.shape)
# %%
for (inp, targ) in train:
    print(inp.shape)
    print(targ.shape)
# %%
inp_tokenizer = data_processer.inp_tokenizer
targ_tokenizer = data_processer.targ_tokenizer
# %%
inp_tokenizer.word_counts
# %%
inp_vocab_size = len(inp_tokenizer.word_counts) + 1
targ_vocab_size = len(targ_tokenizer.word_counts) + 1
# %%
model = TransformerModel(input_vocab_size=inp_vocab_size, target_vocab_size=targ_vocab_size)
# %%
from keras.optimizers import Adam
from transformer.optimizer import CustomLearningRate
# %%
lrate = CustomLearningRate(d_model=512)
optimizer = Adam(learning_rate=lrate)
# %%
trainer = Trainer(model, optimizer=optimizer, epochs=10, checkpoint_folder='./check')
# %%
try:
    trainer.fit(train)
except Exception as e:
    print(str(e))
# %%
# %%
for (inp, targ) in train:
    print(inp[0])


# %%

# %%

# %%

# %%

# %%
