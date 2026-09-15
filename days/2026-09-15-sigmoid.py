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
# No jax.nn.sigmoid -- that is one opaque primitive. Built from arithmetic it
# is four steps, and the animation can only show steps that exist.
@viz(X, palette="neon", size=460, tween=22)
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

# %% [markdown]
# ## As a point cloud
#
# The same formula, but the input is 2-D *coordinates* instead of a field of
# values. Sigmoid is elementwise, so it squashes x and y independently --
# the entire plane folds into the unit square.

# %%
C = jnp.asarray(np.random.default_rng(0).normal(size=(150_000, 2)) * 3.0)

@viz(C, palette="bloom", size=460, tween=22, rep="points")
def sigmoid_points(p):
    return 1 / (1 + jnp.exp(-p))

# %% [markdown]
# ## As graph paper
#
# Now the same transform on a grid, which is where it stops being pretty and
# starts being the reason sigmoid fell out of favour.
#
# Watch the spacing. Near the origin the cells keep their shape -- that is the
# **linear region**. Toward the edges the lines pile up against the boundary:
# a big change in input becomes almost no change in output. That squashing is
# exactly the **vanishing gradient** -- the derivative there is nearly zero, so
# a neuron sitting out here learns almost nothing.

# %%
def _graph_paper(lines=26, per=1100, ext=6.0):
    t = np.linspace(-ext, ext, per)
    L = np.linspace(-ext, ext, lines)
    V = np.stack([np.repeat(L, per), np.tile(t, lines)], 1)
    H = np.stack([np.tile(t, lines), np.repeat(L, per)], 1)
    return np.concatenate([V, H])

GP = jnp.asarray(_graph_paper())

@viz(GP, palette="ice", size=520, tween=26, hold=8, rep="points")
def sigmoid_grid(p):
    return 1 / (1 + jnp.exp(-p))

# %% [markdown]
# ## What I noticed
#
# The range at each step, on an input of -8..8:
#
# | step | min | max |
# |---|---|---|
# | `input` | -8.0000 | 8.00 |
# | `neg` | -8.0000 | 8.00 |
# | `exp` | 0.0003 | **2980.96** |
# | `add` | 1.0003 | 2981.96 |
# | `div` | 0.0003 | 1.00 |
#
# **Which op squashes?** `div`. `exp` blows the range up to nearly 3000 -- four
# orders of magnitude -- and the divide brings all of it back inside 1. The
# animation makes this obvious: the field goes blinding at `exp`, then snaps
# flat.
#
# **Where does the (0, 1) bound come from?** Nothing clips. It falls out of two
# facts about the shape of the expression:
#
# - `exp` is **always positive**, so `1 + exp(-x)` is always `> 1`. One over
#   something bigger than one is always `< 1`. That is the ceiling.
# - as `x -> -inf`, `exp(-x) -> inf`, so the quotient goes to 0 but never
#   reaches it. That is the floor.
#
# So the `+ 1` is not cosmetic -- it is the entire upper bound. Drop it and
# `1/exp(-x)` is just `e^x`, which is unbounded. One character is the
# difference between a probability and an explosion.
#
# Worth noting `exp` hit 2981 on a range of only +-8. At `x = -100` it
# overflows to `inf`, and `1/inf` is 0 -- which is *correct*, and why the naive
# form survives where you might expect NaN. The failure mode is the other
# direction, in the log-likelihood, and that is what `log_sigmoid` exists for.
