# %% [markdown]
# # What a matrix does to a vector
#
# $y = Mx$ is $n^2$ little multiplications, and then they are added up
# and thrown away. The input does not show them. The output does not
# show them. So the one question you actually want answered about a
# linear map — *which input entry was responsible for which part of the
# output?* — is invisible in both vectors.
#
# It lives in the $n^2$ numbers $M_{ij}x_j$. One per wire.
#
# **How to read these.** Inputs are the squares along the bottom,
# outputs the squares along the top, and there is one narrow link for
# every non-zero entry of $M$. A pulse travels each link from the entry
# that sent the contribution to the entry that receives it. Every link
# starts at its own random moment, so you see individual contributions
# landing rather than one curtain coming down — and the output squares
# only take their colour as their pulses arrive.
#
# Colour is value, on one scale shared by the squares and the wires,
# symmetric about zero. The middle of the palette means *nothing here*.
#

# %%
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad

fad.day(__file__)

n = 8
i = jnp.arange(n)
x = jnp.asarray([1.0, 2.0, 1.5, 3.0, 0.8, 2.4, 1.2, 2.8])


# %% [markdown]
# ## A running total
#
# $$M_{ij} = [\,i \ge j\,]$$
#
# Output $0$ gets one wire, output $7$ gets eight. The fan widening
# from left to right *is* the prefix sum — and it is also the reason
# nobody computes one this way: the picture has $O(n^2)$ wires in it.
#

# %%
jaxvis.impact(i[:, None] >= i[None, :], x, name="running_total")


# %% [markdown]
# ## Local averaging
#
# $$M_{ij} = \tfrac{1}{3}\,[\,|i - j| \le 1\,]$$
#
# Three wires per output and all of them short. Nothing travels far,
# which is the whole character of a local operator — and the reason a
# convolution is cheap where the running total was not.
#

# %%
jaxvis.impact((jnp.abs(i[:, None] - i[None, :]) <= 1) / 3.0, x,
              name="local_average")


# %% [markdown]
# ## A permutation
#
# $$M_{ij} = [\,i + j = n - 1\,]$$
#
# Reversal. Eight wires, every one crossing the middle exactly once,
# nothing mixing and nothing lost. One wire in and one wire out of each
# square is what it means for a matrix to only *move* numbers.
#

# %%
jaxvis.impact((i[:, None] + i[None, :] == n - 1) * 1.0, x,
              name="permutation")


# %% [markdown]
# ## A cyclic shift
#
# $$M_{ij} = [\,j = (i + 1) \bmod n\,]$$
#
# A permutation again, but a lazy one: seven wires lean one step over
# and the eighth wraps all the way back across the frame. That one long
# wire is the entire content of the word *cyclic*.
#

# %%
jaxvis.impact((i[None, :] == (i[:, None] + 1) % n) * 1.0, x, name="shift")


# %% [markdown]
# ## A difference
#
# $$M_{ij} = [\,i = j\,] - [\,i - j = 1\,]$$
#
# Two wires per output carrying opposite signs, so the colours split to
# either side of the palette's middle. Every square both sends and
# cancels — which is what makes this the discrete derivative.
#

# %%
D = (jnp.where(i[:, None] == i[None, :], 1, 0)
     + jnp.where(i[:, None] - i[None, :] == 1, -1, 0))

jaxvis.impact(D, x, name="difference")


# %% [markdown]
# ## A learned weight matrix
#
# $$M_{ij} \sim \mathcal{N}(0, 1)$$
#
# No structure at all. Every input reaches every output, the signs go
# both ways, and contributions that cancel arrive at the same square
# from opposite ends of the palette. This is what one layer of a
# trained network looks like, and why reading its weights one entry at
# a time tells you nothing.
#

# %%
W = jnp.asarray(np.random.default_rng(7).normal(size=(n, n)) / 2.4)

jaxvis.impact(W, x, name="dense", spread=0.62)


# %% [markdown]
# ## Try it
#
# Every clip on this page is one line. The matrix can be anything, and
# it does not have to be square:
#
# ```python
# import jaxvis
# jaxvis.impact(M, x, name="mine")
# ```
#
# The knobs: `spread` for how staggered the starts are, `travel` for
# how fast a pulse crosses, `lattice` for how visible the unlit wiring
# is, `palette`, and `seed` for a different random order.
#
# [github.com/aminehd/jaxvis](https://github.com/aminehd/jaxvis)
#
