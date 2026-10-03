import jax.numpy as jnp
import jaxvis

jaxvis.output_to("threads/clips")

FIELD = dict(t=40, extent=1.5, size=420, duration=45, grain=0.06)


@jaxvis.draw_field(palette="ultra", **FIELD)
def a_disc_is_a_distance(x, y, t):
    return jnp.tanh(30 * (t - jnp.sqrt(x ** 2 + y ** 2)))


@jaxvis.draw_field(palette="bloom", **FIELD)
def ripples(x, y, t):
    return jnp.sin(14 * jnp.sqrt(x ** 2 + y ** 2) - 6.3 * t)


@jaxvis.draw_field(palette="vapor", **FIELD)
def a_pinwheel(x, y, t):
    return jnp.sin(6 * jnp.arctan2(y, x) + 9 * jnp.sqrt(x ** 2 + y ** 2) - 6.3 * t)
