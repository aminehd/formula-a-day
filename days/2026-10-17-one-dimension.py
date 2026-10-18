# %% [markdown]
# # One dimension
#
# A point cloud shows you where things go. A curve shows you what a
# function *is*. Same machinery underneath: every line below is one
# jaxpr, drawn one primitive at a time, so each frame is a real
# intermediate value and not an illustration of one.
#
# Each clip starts as a flat line at zero, rises into the input
# $y = x$, and from there every bend is one more primitive. The
# caption under the frame names the one you are looking at.
#
# The window is fixed, so a curve that leaves it is pinned dimly to
# the edge rather than disappearing -- an exponential on its way up
# is still somewhere.
#

# %%
import jax.numpy as jnp
import jaxvis
import fad

fad.day(__file__)

x = jaxvis.rail(-6.0, 6.0, 500)


# %% [markdown]
# ## The squash
#
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
#
# Four primitives: negate, exponentiate, add one, divide. The straight
# sloping line early on is $-x$ -- negation is a reflection, and a
# reflected straight line is still a straight line. Then the
# exponential blows up off the top of the frame, and the division
# folds it back into a window one unit tall.
#

# %%
@jaxvis.draw(x, **jaxvis.curve(x, ylim=(-0.35, 1.35), palette="vapor"))
def sigmoid(p):
    return 1.0 / (1.0 + jnp.exp(-p))


# %% [markdown]
# ## The soft hinge
#
# $$\operatorname{softplus}(x) = \log(1 + e^{x})$$
#
# The same exponential, bent the other way. The log is what turns a
# wall into a hinge.
#

# %%
@jaxvis.draw(x, **jaxvis.curve(x, ylim=(-1.0, 6.5), palette="ultra",
                               fill=0.08))
def softplus(p):
    return jnp.log1p(jnp.exp(p))


# %% [markdown]
# ## GELU
#
# $$\operatorname{gelu}(x) \approx \tfrac{1}{2}x\left(1 +
#   \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715x^{3}\right)\right]\right)$$
#
# The activation inside almost every transformer you will ever profile.
# It is a sigmoid-shaped gate multiplied back onto $x$ — and the dip
# just left of zero, the part a ReLU does not have, arrives in the very
# last multiply.
#

# %%
@jaxvis.draw(x, **jaxvis.curve(x, ylim=(-1.5, 6.5), palette="bloom"))
def gelu(p):
    inner = 0.7978845608 * (p + 0.044715 * p ** 3)
    return 0.5 * p * (1.0 + jnp.tanh(inner))


# %% [markdown]
# ## A square out of sines
#
# $$f(t) = \sin t + \tfrac{1}{3}\sin 3t + \tfrac{1}{5}\sin 5t +
#   \tfrac{1}{7}\sin 7t$$
#
# On a fixed window the corrections are finally visible: each harmonic
# arrives at full height, gets divided down, and settles into the
# corners. A window that rescales itself every frame would hide exactly
# that — the division would animate as nothing happening at all.
#

# %%
s = jaxvis.rail(-jnp.pi, jnp.pi, 500)


@jaxvis.draw(s, **jaxvis.curve(s, ylim=(-1.45, 1.45), palette="vapor"))
def square_wave(p):
    y = jnp.sin(p)
    y = y + jnp.sin(3.0 * p) / 3.0
    y = y + jnp.sin(5.0 * p) / 5.0
    return y + jnp.sin(7.0 * p) / 7.0


# %% [markdown]
# ## The bell
#
# $$y = e^{-x^{2}/2}$$
#
# Square, negate, halve, exponentiate. The parabola that falls off the
# bottom of the frame comes back as the only bounded shape in the file.
#

# %%
@jaxvis.draw(x, **jaxvis.curve(x, ylim=(-0.3, 1.25), palette="ember",
                               fill=0.10))
def bell(p):
    return jnp.exp(-(p ** 2) / 2.0)


# %% [markdown]
# ## Ringing down
#
# $$y = e^{-t/3}\sin 4t$$
#
# A sine multiplied by a decay. The envelope never gets drawn — it is
# the one multiply at the end, and you see it as the whole curve being
# pulled toward the axis from the right.
#

# %%
r = jaxvis.rail(0.0, 12.0, 500)


@jaxvis.draw(r, **jaxvis.curve(r, ylim=(-1.15, 1.15), palette="ultra"))
def ring_down(p):
    return jnp.exp(-p / 3.0) * jnp.sin(4.0 * p)
