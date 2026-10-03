import jax.numpy as jnp
import numpy as np
import jaxvis

jaxvis.output_to("threads/clips")

n = 40_000
rng = np.random.default_rng(0)
bar = jnp.asarray(rng.uniform(-1, 1, size=(n, 2)) * np.array([2.2, 0.45]))
angle = np.arctan2(np.asarray(bar)[:, 1], np.asarray(bar)[:, 0])
hue = np.stack([np.sin(angle), np.sin(angle + 2.1), np.sin(angle + 4.2)], 1) ** 2


@jaxvis.draw(bar, rep={(n, 2): "points"}, only="points", structural=True,
             size=640, tween=30, hold=8, duration=55, palette="bloom",
             colors=hue, blend="mean", frame=(0.0, 0.0, 2.6))
def rotate(x):
    t = 1.0
    r = jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
    return x @ r.T
