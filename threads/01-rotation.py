import jax.numpy as jnp
import jaxvis
from jaxvis import data, tint

jaxvis.output_to("threads/clips")

bar = data.bar(40_000)
t = 1.0


@jaxvis.draw(bar, colors=tint.by_angle(bar), frame=(0.0, 0.0, 2.6),
             **jaxvis.cloud(bar))
def rotate(x):
    r = jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
    return x @ r.T
