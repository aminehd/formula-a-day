# %% [markdown]
# # Building blocks
#
# Six compositions, each built from pieces that take a point in the
# plane and hand back a point in the plane. Every step is named after
# the operation that made it, so you can read what happened.
#
# <!-- page: fun -->

# %%
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad
from jaxvis import tint

fad.day(__file__)

n = 1600
half = jnp.pi / 2


def clock(lo, hi, lag=half):
    t = jnp.linspace(lo, hi, n)[:, None]
    return t, jnp.concatenate([t, t - lag], 1)


def sheet(m=60_000, extent=2.2, seed=0):
    r = np.random.default_rng(seed)
    return jnp.asarray(r.uniform(-extent, extent, (m, 2)), dtype=jnp.float32)


def mesh(k=130, extent=2.2):
    u = np.linspace(-extent, extent, k)
    return jnp.asarray(np.stack(np.meshgrid(u, u), -1).reshape(-1, 2),
                       dtype=jnp.float32)


def hue(t, palette):
    return tint.by_value(np.asarray(t).ravel(), palette)


# %% [markdown]
# ## Euler
#
# $$e^{i\theta} = \cos\theta + i\sin\theta$$
#
# One piece. A slanted line goes in, a circle comes out, because
# $\sin$ is $\cos$ running a quarter turn late.
#

# %%
t, lagged = clock(-jnp.pi, jnp.pi)


@jaxvis.draw(lagged, colors=hue(t, "vapor"),
             **jaxvis.cloud(lagged, palette="vapor"))
def euler(p):
    return jnp.cos(p)


# %% [markdown]
# ## Damped oscillation
#
# $$z = e^{-t/\tau}\,e^{i\omega t}$$
#
# The circle again, then multiplied by an envelope that shrinks.
# A spiral is just a circle losing its radius.
#

# %%
d, dlag = clock(0.0, 16.0, half / 2)


@jaxvis.draw(dlag, colors=hue(d, "bloom"), frame=(0.0, 0.0, 1.2),
             **jaxvis.cloud(dlag, palette="bloom"))
def damped(p):
    ring = jnp.cos(2.0 * p)
    return ring * jnp.exp(-p[:, :1] / 7.0)


# %% [markdown]
# ## Lissajous
#
# $$x = \sin 3t, \qquad y = \sin\!\left(2t - \tfrac{\pi}{2}\right)$$
#
# Scale the clock differently in each direction, then take one sine.
# The ratio decides the knot.
#

# %%
l, llag = clock(0.0, 2 * jnp.pi)


@jaxvis.draw(llag, colors=hue(l, "vapor"),
             **jaxvis.cloud(llag, palette="vapor"))
def lissajous(p):
    return jnp.sin(p * jnp.array([[3.0, 2.0]]))


# %% [markdown]
# ## A square out of sines
#
# $$f(t) = \sin t + \tfrac{1}{3}\sin 3t + \tfrac{1}{5}\sin 5t + \tfrac{1}{7}\sin 7t$$
#
# The square wave, drawn against a quarter-turn-late copy of itself.
# Each harmonic pushes the corners further out.
#

# %%
s, slag = clock(-jnp.pi, jnp.pi)


@jaxvis.draw(slag, colors=hue(s, "ultra"), **jaxvis.cloud(slag))
def square_wave(p):
    y = jnp.sin(p)
    y = y + jnp.sin(3.0 * p) / 3.0
    y = y + jnp.sin(5.0 * p) / 5.0
    return y + jnp.sin(7.0 * p) / 7.0


# %% [markdown]
# ## The bell
#
# $$y = e^{-x^2}$$
#
# A sheet of points, squared, flipped, exponentiated. Every point is
# dragged toward the corner, and the crowding is the shape of the bell.
#

# %%
g = sheet()
w = mesh()


@jaxvis.draw(w, colors=hue(w[:, 0], "ember"), **jaxvis.cloud(w, palette="ember"))
def the_bell(p):
    return jnp.exp(-(p ** 2))


# %% [markdown]
# ## The squash
#
# $$\tanh(kx)$$
#
# The same sheet, pressed into a square. This is the saturation that
# every activation function is trying to buy.
#

# %%


@jaxvis.draw(g, colors=tint.by_radius(g), **jaxvis.cloud(g, palette="bloom"))
def the_squash(p):
    return jnp.tanh(1.6 * p)
