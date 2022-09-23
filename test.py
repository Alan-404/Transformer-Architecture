#%%
from preprocessing.process_data import Data
from transformer.model import TransformerModel
from transformer.trainer import Transformer
# %%
data_processer = Data('en', 'vi', './check')
# %%
train, val = data_processer.build_dataset('./datasets/en_sents.txt', './datasets/vi_sents.txt', buffer_size=64, batch_size=64, max_length=40, num_data=2000)
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
model = Transformer(input_vocab_size=inp_vocab_size, target_vocab_size=targ_vocab_size)
# %%

# %%
my_model = model.fit(train, epochs=10)
# %%
# %%
# %%

# %%

# %%

# %%

# %%

# %%
