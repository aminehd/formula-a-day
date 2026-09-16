# %% [markdown]
# # Tanh
#
# $$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$
#
# Sigmoid's cousin. Same S-shape, but centred on zero and spanning -1 to 1
# instead of 0 to 1.

# %%
import jax.numpy as jnp
import numpy as np
import fad
import jaxvis

fad.day(__file__)

# Rings that fade outward -- bounded, and varying in both axes, so no step
# lands in exp's extremes and renders black.
_g = jnp.linspace(-1, 1, 140)
_YY, _XX = jnp.meshgrid(_g, _g, indexing="ij")
_R = jnp.hypot(_XX, _YY)
# sin(4R) is what makes it RINGS -- without it this is one smooth bump.
# (1 - R) fades them outward so nothing reaches tanh's flat extremes.
X = 3 * jnp.sin(4 * _R) * (1 - _R)


@jaxvis.draw(X, palette="ultra", size=680, tween=30, hold=8, duration=55,
     rep_kw={"grain": 0.26})
def tanh(x):
    return jnp.tanh(x)

# %% [markdown]
# ## As graph paper
#
# The input is 2-D *coordinates* now, not a field of values. tanh is
# elementwise, so it squashes x and y independently and the whole plane folds
# into the square from -1 to 1.
#
# Watch the spacing: cells keep their shape near the origin (the linear
# region) and pile up against the edges (saturation). Unlike sigmoid, tanh is
# centred on zero -- the origin stays put instead of moving to 0.5.

# %%
def _graph_paper(lines=26, per=1100, ext=3.0):
    "Points sampled ALONG grid lines -- the point renderer draws them as paper."
    t = np.linspace(-ext, ext, per)
    L = np.linspace(-ext, ext, lines)
    V = np.stack([np.repeat(L, per), np.tile(t, lines)], 1)   # vertical lines
    H = np.stack([np.tile(t, lines), np.repeat(L, per)], 1)   # horizontal
    return np.concatenate([V, H])

GP = jnp.asarray(_graph_paper())

@jaxvis.draw(GP, palette="ice", size=760, tween=30, hold=8, duration=55, rep="points")
def tanh_grid(p):
    return jnp.tanh(p)

# %% [markdown]
# ## The grid, coloured by where it came from
#
# Same deformation, but each point is coloured by its **original position**
# rather than by how many particles landed on the pixel. Red rises with x,
# green with y, so every point carries a label saying where it started.
#
# `colors=` is the only new thing: give `@viz` one RGB per point and it routes
# the point steps through the colour renderer instead of the density one.

# %%
_SRC = np.asarray(_graph_paper(lines=26, per=1100, ext=3.0))
_U = (_SRC + 3.0) / 6.0                              # position -> 0..1

def _vivid(rgb):
    "Keep the hue, push it to full value."
    return rgb / (rgb.max(1, keepdims=True) + 1e-9)

C_POS = _vivid(np.stack([_U[:, 0], _U[:, 1], 1 - _U[:, 0]], 1))

@jaxvis.draw(GP, palette="ice", size=700, tween=30, hold=8, duration=55,
     rep="points", colors=C_POS)
def tanh_rainbow(p):
    return jnp.tanh(p)

# %% [markdown]
# ## Coloured in and out
#
# Coloured by **how far from the centre each point started** — inner rings one
# colour, outer rings another, straight off the `ultra` palette. `ramp()` was
# written for heatmaps, but it is only a 256x3 lookup table, so it works just
# as well as a source of point colours.

# %%
from jaxvis.render import ramp

_R_SRC = np.hypot(_SRC[:, 0], _SRC[:, 1])
_UR = np.clip(_R_SRC / _R_SRC.max(), 0, 1)
C_RADIUS = _vivid(ramp("ultra")[(_UR * 255).astype(int)] / 255.0)

@jaxvis.draw(GP, palette="ice", size=700, tween=30, hold=8, duration=55,
     rep="points", colors=C_RADIUS)
def tanh_inout(p):
    return jnp.tanh(p)

# %% [markdown]
# ## Stripes
#
# The radius chopped into eight hard bands instead of a smooth ramp. A gradient
# you have to squint at; stripes you can **count**.
#
# They start equal width. By the end the inner ones are still wide and the
# outer ones are slivers against the edge — and because they are discrete you
# can see which band lost the most. That ratio is the derivative of tanh, read
# off the picture.

# %%
STRIPE = np.array([                  # eight distinct colours, no blending
    [255,  60, 120], [255, 160,  40], [250, 240,  70], [120, 255, 120],
    [ 60, 230, 200], [ 70, 150, 255], [160, 110, 255], [255, 120, 220],
], dtype=float) / 255.0

C_STRIPE = STRIPE[np.floor(np.clip(_UR, 0, 0.999) * len(STRIPE)).astype(int)]

@jaxvis.draw(GP, palette="ice", size=700, tween=30, hold=8, duration=55,
     rep="points", colors=C_STRIPE)
def tanh_stripes(p):
    return jnp.tanh(p)

# %% [markdown]
# ## Circular paper
#
# Polar lines: concentric rings and radial spokes, coloured by angle so every
# spoke keeps its hue through the deformation.
#
# This is the natural paper for tanh, because tanh acts on x and y separately
# and circles are the shape with no preferred axis. A circle does not stay a
# circle — it flattens toward a square, because along a diagonal you must go
# a factor of sqrt(2) further out to saturate as hard as along an axis.

# %%
def _polar_paper(rings=16, spokes=24, per=1400, ext=3.0):
    "Concentric circles + radial spokes."
    th = np.linspace(0, 2 * np.pi, per)
    R = np.linspace(ext / rings, ext, rings)
    circles = np.concatenate([np.stack([r * np.cos(th), r * np.sin(th)], 1)
                              for r in R])
    rr = np.linspace(0, ext, per)
    A = np.linspace(0, 2 * np.pi, spokes, endpoint=False)
    rays = np.concatenate([np.stack([rr * np.cos(a), rr * np.sin(a)], 1)
                           for a in A])
    return np.concatenate([circles, rays])

_POLAR = _polar_paper()
PP = jnp.asarray(_POLAR)
_H = (np.arctan2(_POLAR[:, 1], _POLAR[:, 0]) + np.pi) / (2 * np.pi)
C_ANGLE = _vivid(np.stack([np.sin(np.pi * _H), np.sin(np.pi * (_H + 1/3)),
                           np.sin(np.pi * (_H + 2/3))], 1) ** 2)

@jaxvis.draw(PP, palette="ice", size=700, tween=30, hold=8, duration=55,
     rep="points", colors=C_ANGLE)
def tanh_polar(p):
    return jnp.tanh(p)

# %% [markdown]
# ## Straight stripes
#
# The ringed field depends on `_R`, the distance from the centre, so its bands
# come out circular. Make the value depend on **x alone** and the bands run
# straight top to bottom — the geometry lives entirely in what the value is a
# function of.
#
# `sin` gives soft-edged stripes; `%` (modulo) gives hard ones. Soft stripes
# show tanh flattening the tops and bottoms of each wave while the steep parts
# through zero survive — the squash, seen edge-on.

# %%
# _XX varies left-to-right and _YY top-to-bottom, so a function of _XX alone
# is constant down each column: vertical stripes.
XS = 3 * jnp.sin(6 * _XX)

@jaxvis.draw(XS, palette="ultra", size=680, tween=30, hold=8, duration=55,
     rep_kw={"grain": 0.26})
def tanh_stripe_field(x):
    return jnp.tanh(x)

# %% [markdown]
# And with modulo instead, for hard edges — a sawtooth rather than a wave.

# %%
XM = 6 * ((_XX % 0.5) - 0.25)

@jaxvis.draw(XM, palette="spectral", size=680, tween=30, hold=8, duration=55,
     rep_kw={"grain": 0.26})
def tanh_sawtooth(x):
    return jnp.tanh(x)
