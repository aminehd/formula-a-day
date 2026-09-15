## Just because they're beautiful

Same framework, no lesson attached.

- **shear** — each axis displaced by the sine of the other. 200k particles.

```python
flip = jnp.concatenate([p[:, 1:], p[:, :1]], 1)    # (y, x)
p + 1.3 * jnp.sin(1.7 * flip)
```

- **fold** — sine domain folding against a rotation.
- **galaxy** — rotate by `1/r`, so the inside spins faster than the outside. That
  is why real spiral galaxies have arms.
- **petals** — rotation by `sin(5θ)`, five-fold.
- **liquid_flow** — domain warping: the field read at coordinates pushed around
  by a drifting noise field.
- **projection** — 96 dimensions of noise, fitted down onto a 2-simplex. The
  triangle was always in there.
