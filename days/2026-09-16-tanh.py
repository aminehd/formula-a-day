# %% [markdown]
# # Tanh
#
# $$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$
#
# _Write-up goes here._

# %%
import jax.numpy as jnp
import numpy as np
import fad
import jaxvis

fad.day(__file__)

# A grid of coordinates, used by everything below.
# grid_x runs left to right, grid_y runs top to bottom.
axis = jnp.linspace(-1, 1, 140)
grid_y, grid_x = jnp.meshgrid(axis, axis, indexing="ij")

# %% [markdown]
# ## Random points
#
# _Notes._

# %%
rng = np.random.default_rng(0)
cloud = jnp.asarray(rng.uniform(-2.5, 2.5, size=(150_000, 2)))


@jaxvis.draw(cloud, palette="bloom", size=680, tween=30, hold=8, duration=55,
             rep="points")
def tanh_cloud(p):
    return jnp.tanh(p)


# %% [markdown]
# ## sin 
#
# _Notes._
#
# Now sinus
# %%
@jaxvis.draw(cloud, palette="bloom", size=680, tween=30, hold=8, duration=55)
def sin_cloud(p):
    return jnp.sin(p)


# %% [markdown]
# ## As a field
#
# _Notes._

# %%
# A lattice, not rings. `%` wraps each axis into repeating cells; subtracting
# half a cell centres each one on zero, so every cell runs negative -> positive.
# Multiplying the two gives a grid of alternating sign -- a checkerboard of
# hills and valleys, spanning about -3..3.
cell = 0.35
bars_x = (grid_x % cell) / cell - 0.5          # sawtooth across x
bars_y = (grid_y % cell) / cell - 0.5          # sawtooth down y
field = 12 * bars_x * bars_y


@jaxvis.draw(field, palette="ultra", size=680, tween=30, hold=8, duration=55,
             rep_kw={"grain": 0.26})
def tanh(x):
    return jnp.tanh(x)



# %% [markdown]
# ##  Lets just multiply
#
# _Notes._

# %%
@jaxvis.draw(field, palette="ultra", size=680, tween=30,
             hold=8, duration=55)
def multiply(x):
    return 20 *x + 4

# %% [markdown]
# ## As points
#
# _Notes._

# %%
def graph_paper(lines=26, per=1100, extent=3.0):
    "Points sampled ALONG grid lines -- the point renderer draws them as paper."
    along = np.linspace(-extent, extent, per)
    at = np.linspace(-extent, extent, lines)
    vertical = np.stack([np.repeat(at, per), np.tile(along, lines)], 1)
    horizontal = np.stack([np.tile(along, lines), np.repeat(at, per)], 1)
    return np.concatenate([vertical, horizontal])


paper = jnp.asarray(graph_paper())


@jaxvis.draw(paper, palette="ice", size=700, tween=30, hold=8, duration=55,
             rep="points")
def tanh_grid(p):
    return jnp.tanh(p)


# %% [markdown]
# ## Coloured in and out
#
# _Notes._

# %%
from jaxvis.render import ramp

paper_np = np.asarray(paper)
radius = np.hypot(paper_np[:, 0], paper_np[:, 1])
depth = np.clip(radius / radius.max(), 0, 1)        # 0 at centre, 1 at the edge
by_radius = ramp("ultra")[(depth * 255).astype(int)] / 255.0
by_radius = by_radius / (by_radius.max(1, keepdims=True) + 1e-9)


# colors= paints each point its own RGB at full value, instead of dimming it by
# how many particles hit the pixel -- which is why these lines read sharper.
@jaxvis.draw(paper, palette="ice", size=700, tween=30, hold=8, duration=55,
             rep="points", colors=by_radius)
def tanh_inout(p):
    return jnp.tanh(p)

# %% [markdown]
# ## just try sinus 
#
# _Notes._

# %%
@jaxvis.draw(paper, palette="ice", size=700, tween=30, hold=8, duration=55,
             rep="points")
def sin_grid(p):
    return jnp.sin(p)

