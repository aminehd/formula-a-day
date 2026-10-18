# %% [markdown]
# # The puzzles, as wiring
#
# Every Tensor Puzzle I have solved so far came down to the same move:
# stop thinking about a loop, and build a matrix instead. The recipe is
# two sentences.
#
# > **Row $i$** says what builds output $i$.
# > **Column $j$** says where input $j$ goes.
#
# Write the rule in $i$ and $j$, put the row-thing in `[:, None]` and
# the column-thing in `[None, :]`, let broadcasting make the grid, and
# the matrix is done. Then one `@`.
#
# What the recipe does not show me is the *movement*. So here it is:
# inputs along the bottom, outputs along the top, one narrow link for
# every non-zero entry of the matrix, and a pulse travelling from the
# entry that sent it to the entry that receives it. Each link starts at
# its own moment, so I can watch individual contributions land instead
# of one curtain coming down.
#
# The colour of a square is its value. The colour of a wire is the
# contribution it is carrying, $M_{ij}x_j$, on the same scale.
#

# %%
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad

fad.day(__file__)

n = 8
i = jnp.arange(n)
a = jnp.asarray([1.0, 2.0, 1.5, 3.0, 0.8, 2.4, 1.2, 2.8])


# %% [markdown]
# ## p07 — cumsum
#
# $$y_i = \sum_{j \le i} a_j
#   \qquad\Longleftrightarrow\qquad
#   M_{ij} = [\,i \ge j\,]$$
#
# ```python
# def cumsum(a):
#     return (arange(a.shape[0])[:, None] >= arange(a.shape[0])[None, :]) @ a
# ```
#
# Output $0$ gets one wire. Output $7$ gets eight. The fan widening left
# to right *is* the running total.
#
# This is the one that was silently wrong for three weeks, because I had
# written `<=` instead of `>=`. Flipped, the wiring leans the other way
# and the answer is a suffix sum — right shape, plausible numbers, no
# error. Worth remembering what a wrong matrix looks like.
#

# %%
jaxvis.impact(i[:, None] >= i[None, :], a, name="cumsum")


# %% [markdown]
# ## p08 — diff
#
# $$y_i = a_i - a_{i-1}
#   \qquad\Longleftrightarrow\qquad
#   M_{ij} = [\,i = j\,] - [\,i - j = 1\,]$$
#
# ```python
# def diff(a, i):
#     return (where(arange(len(a))[:, None] == arange(len(a))[None, :], 1, 0)
#             + where(arange(len(a))[:, None] - arange(len(a))[None, :] == 1, -1, 0)) @ a
# ```
#
# Two wires per output and they carry opposite signs — the colours split
# to either side of the palette's middle. Every wire is short, which is
# what local means.
#

# %%
D = (jnp.where(i[:, None] == i[None, :], 1, 0)
     + jnp.where(i[:, None] - i[None, :] == 1, -1, 0))

jaxvis.impact(D, a, name="diff")


# %% [markdown]
# ## p11 — flip
#
# $$y_i = a_{n-1-i}
#   \qquad\Longleftrightarrow\qquad
#   M_{ij} = [\,i + j = n - 1\,]$$
#
# ```python
# def flip(a, i):
#     return a[i - 1 - arange(i)]
# ```
#
# I solved this one by indexing, not with a matrix. The wiring shows
# they are the same move seen from two sides: eight wires, every one
# crossing the middle exactly once, nothing mixing. A permutation.
#

# %%
jaxvis.impact((i[:, None] + i[None, :] == n - 1) * 1.0, a, name="flip")


# %% [markdown]
# ## p10 — roll
#
# $$y_i = a_{(i+1) \bmod n}
#   \qquad\Longleftrightarrow\qquad
#   M_{ij} = [\,j = (i+1) \bmod n\,]$$
#
# ```python
# def roll(a, i):
#     return a[where(arange(i) + 1 < i, arange(i) + 1, 0)]
# ```
#
# A permutation again, but a lazy one: seven wires lean one step to the
# left and the eighth wraps all the way back. That long wire is the
# whole content of the word *cyclic*.
#

# %%
jaxvis.impact((i[None, :] == (i[:, None] + 1) % n) * 1.0, a, name="roll")


# %% [markdown]
# ## p12 — compress
#
# $$y = (g \odot v)\,M, \qquad M_{j k} = [\,\operatorname{cumsum}(g)_j - 1 = k\,]$$
#
# ```python
# def compress(g, v, i):
#     return (g * v) @ (cumsum(g * 1)[:, None] - 1 == arange(g.shape[0])[None, :])
# ```
#
# The interesting one. The matrix is not fixed — it is **built from the
# data**, because where each kept value lands depends on how many were
# kept before it. The prefix sum of the mask is the destination slot.
#
# Only the kept entries get a wire at all. The dropped ones are the
# masked squares on the bottom with nothing leaving them, and the empty
# slots on the top are the tail that nothing reaches.
#
# This is the shape of the whole trick: a rearrangement that depends on
# the data, with a shape that does not. It is how a GPU filters.
#

# %%
g = jnp.asarray([1, 0, 1, 1, 0, 1, 0, 0])
cs = (i[:, None] >= i[None, :]) @ g                # p07, reused
C = (cs[:, None] - 1 == i[None, :]) * 1.0

# y = (g*v) @ C  is the same as  C.T @ (g*v)
jaxvis.impact(C.T, g * a, name="compress", label="y = (g*v) M   scatter")


# %% [markdown]
# ## p13 — pad_to
#
# $$\texttt{pad\_to}(a, i, j): \quad i \text{ numbers in}, \; j \text{ slots out}$$
#
# The next one, and the first where **the output is not the same length
# as the input**. So the matrix stops being square, and the recipe has
# to say something about the rows that have no input to draw from and
# the inputs that have no row to land in.
#
# Here is the question and nothing else: five slots in, eight out, no
# wiring yet. Working out which entries are ones is the puzzle, and I
# would rather find it than be shown it.
#

# %%
jaxvis.impact(jnp.zeros((8, 5)), a[:5], name="pad_to",
              label="p13 pad_to   (8, 5) @ (5,)   -- unsolved")
