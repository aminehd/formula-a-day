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
import fad
import jaxvis

fad.day(__file__)

# Rings that fade outward. A plain ramp would put most of the field in exp's
# extremes, and since the heatmap normalises to the max, the rest renders
# black. Keeping the values bounded and varying in BOTH axes keeps every step
# readable -- worst-lit frame goes from 16% of the canvas to 67%.
_g = jnp.linspace(-1, 1, 140)
_YY, _XX = jnp.meshgrid(_g, _g, indexing="ij")
_R = jnp.hypot(_XX, _YY)
X = 3 * jnp.sin(4 * _R) * (1 - _R)

# %% [markdown]
# ## As a field
#
# Every intermediate value, animated. Watch the range: `exp` explodes it, then
# something pulls it back into (0, 1). Nothing in the chain says "clip" -- the
# bound is a consequence.

# %%
# No jax.nn.sigmoid -- that is one opaque primitive. Built from arithmetic it
# is four steps, and the animation can only show steps that exist.
# grain: fine static texture inside as_heatmap, strongest in the midtones and
# absent at pure black/white (ink on fibre). A smooth bicubic gradient reads
# as glossy plastic; the point clouds look matte because they are made of
# discrete splatted particles. This puts that texture back.
# Bumped to 0.26 because the render is downscaled 680 -> ~420 on the page,
# which averages fine grain away.
@jaxvis.draw(X, palette="ultra", size=680, tween=30, hold=8, duration=55,
     rep_kw={"grain": 0.26})
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

# %% [markdown]
# ## From noise
#
# Same formula, random input. Nothing is designed here -- and that is the point.
# The input is uniform noise over -3.5..3.5, and sigmoid pulls all of it into
# 0.03..0.97, crowding most values toward the two ends. Push the range wider and
# it becomes a **step function**.
#
# That is saturation again, from a third angle: the grid showed it as spacing,
# the cloud as a boundary, and this shows it as *lost information*. Two inputs
# of 6 and 9 are very different numbers and come out almost identical.

# %%
# Uniform, not Gaussian -- same lesson as the point cloud. normal*6 reaches
# +-25, and exp(25) is 7e10, so a few pixels blacken the whole frame.
N = jnp.asarray(np.random.default_rng(3).uniform(-3.5, 3.5, size=(140, 140)))

@jaxvis.draw(N, palette="vapor", size=680, tween=30, hold=8, duration=55,
     rep_kw={"grain": 0.26})
def sigmoid_noise(x):
    return 1 / (1 + jnp.exp(-x))

# %% [markdown]
# ## As a point cloud
#
# The same formula, but the input is 2-D *coordinates* instead of a field of
# values. Sigmoid is elementwise, so it squashes x and y independently --
# the entire plane folds into the unit square.
#
# The input is a **uniform square, not a Gaussian**, and that matters more than
# it looks. A Gaussian has tails: with `normal * 3` a few of the 150k points sit
# at +-14, and `exp(14)` is ~1.2 million. Those few points then set the scale
# for every frame, so the other 149,990 collapse into a corner. Measured share
# of the canvas actually lit at the worst step: `normal * 3` gives 0.1%, a
# disc of radius 3 gives 10.1%, a uniform square gives **23.1%**.
#
# Bounded input, no tails, nothing to blow up. `exp` punishes outliers
# exponentially -- which is the thing to remember well beyond this plot.

# %%
C = jnp.asarray(np.random.default_rng(0).uniform(-2.5, 2.5, size=(150_000, 2)))

# tween = interpolation frames per step; duration = ms per frame.
# Together they set how long you get to actually watch each op.
@jaxvis.draw(C, palette="bloom", size=680, tween=32, rep="points",
     hold=8, duration=55)
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

@jaxvis.draw(GP, palette="ice", size=760, tween=30, hold=8, duration=55,
     rep="points")
def sigmoid_grid(p):
    return 1 / (1 + jnp.exp(-p))

# %% [markdown]
# ## What I noticed
#
# Watching the range at each step: `input` and `neg` both span -8 to 8, `exp`
# blows it out to nearly **3000**, `add` shifts that to 1..2982, and `div`
# brings all of it back to 0..1.
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
