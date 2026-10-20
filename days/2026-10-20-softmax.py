# %% [markdown]
# # Softmax, and the subtraction nobody mentions
#
# Twenty-four scores go in, twenty-four probabilities come out. The
# textbook writes two operations. The jaxpr shows four.
#
# <!-- order: 6 -->

# %%
import jax.numpy as jnp
import jaxvis
import fad

fad.day(__file__)

x = jaxvis.rail(-6.0, 6.0, 24)


# %% [markdown]
# ## The formula as written
#
# $$\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
#
# Exponentiate, then divide by the total. Watch the sum arrive as a flat
# line: one number, broadcast back across all twenty-four.
#

# %%
@jaxvis.draw(x, **jaxvis.curve(x, ylim=(-0.15, 1.15), palette="vapor"))
def softmax(p):
    e = jnp.exp(p)
    return e / jnp.sum(e)


# %% [markdown]
# ## The formula as implemented
#
# $$\operatorname{softmax}(x)_i = \frac{e^{x_i - \max x}}{\sum_j e^{x_j - \max x}}$$
#
# Subtracting the largest score changes nothing — it cancels, top and
# bottom. What it buys is that the biggest exponent is now $e^0 = 1$, so
# nothing can overflow. Two extra steps in the jaxpr, and the curve that
# comes out is identical.
#

# %%
@jaxvis.draw(x, **jaxvis.curve(x, ylim=(-0.15, 1.15), palette="bloom"))
def stable_softmax(p):
    e = jnp.exp(p - jnp.max(p))
    return e / jnp.sum(e)
