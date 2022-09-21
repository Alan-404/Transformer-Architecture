#%%
from transformer.behaviors.mask import generate_padding_mask
# %%
import numpy as np
# %%
a = np.array([[1,2,3], [1,3,4], [2,3,4]])
# %%
a.shape
# %%
test = generate_padding_mask(a)
# %%
test.shape
# %%
from transformer.behaviors.positional_encoding import encode_position
# %%
en = encode_position(length=12, d_model=512)
# %%
en
# %%
en.shape
# %%

# %%

# %%

# %%
