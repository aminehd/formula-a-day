import numpy as np
import jaxvis
from jaxvis import data, tint

jaxvis.output_to("threads/clips")

x = data.tilted(40_000)
mu = np.asarray(x).mean(0)
unmix = np.linalg.inv(np.linalg.cholesky(np.cov(np.asarray(x).T))).T


@jaxvis.draw(x, colors=tint.by_angle(x), frame="fixed", **jaxvis.cloud(x))
def whiten(p):
    return (p - mu) @ unmix
