# %% [markdown]
# # Sigmoid
#
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
#
# The squashing function. Takes any real number and returns something strictly
# between 0 and 1, which is why it reads as a probability.
#
# **Write your notes here.** What surprised you?

# %%
import jax.numpy as jnp
import numpy as np
from fad import viz

X = jnp.linspace(-8, 8, 160)[:, None] * jnp.ones((1, 160))

# %% [markdown]
# ## As a field
#
# Every intermediate value, animated. Watch the range: `exp` explodes it, then
# something pulls it back into (0, 1). Nothing in the chain says "clip" -- the
# bound is a consequence.

# %%
# TODO: one line, no jax.nn -- build it from arithmetic.
@viz(X, palette="neon", size=460, tween=22)
def sigmoid(x):
    raise NotImplementedError

# %% [markdown]
# ## As a curve
#
# The same formula on a 1-D sweep, so you can see the S.

# %%
T = jnp.linspace(-8, 8, 4000)

@viz(T, palette="ice", size=460, tween=22)
def sigmoid_curve(x):
    raise NotImplementedError

# %% [markdown]
# ## What I noticed
#
# - which op does the squashing?
# - where does the (0, 1) bound come from?
