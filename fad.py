"""One file per day. Write markdown + code, run it, get a page.

    from fad import viz

    @viz(X, palette="neon")
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

`viz` is `jaxvis.visualize` with the output path decided for you: it lands in
site/<this file's slug>/, which is exactly what gets published.
"""
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _slug_of_caller():
    for fr in inspect.stack()[1:]:
        p = Path(fr.filename)
        if p.parent.name == "days":
            return p.stem
    return "scratch"


def viz(*args, name=None, **kw):
    """Render a function into this day's folder. Same kwargs as visualize()."""
    slug = _slug_of_caller()
    d = ROOT / "site" / slug
    d.mkdir(parents=True, exist_ok=True)

    def deco(fn):
        from jaxvis import visualize
        tag = name or fn.__name__
        kw.setdefault("verbose", False)
        try:
            visualize(fn, *args, out=str(d / f"{tag}.gif"), **kw)
            print(f"  {slug}/{tag}.gif")
        except Exception as e:
            print(f"  x {tag}: {type(e).__name__}: {e}")
        return fn
    return deco
