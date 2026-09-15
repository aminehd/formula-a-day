# a formula a day

One ML building block per day: the formula, what it does, and every
intermediate value animated with [jaxvis](../jaxvis-pkg).

```
days/YYYY-MM-DD-name.py    write here -- markdown cells + code + @viz
docs/                     generated; this is what GitHub Pages serves
```

```bash
python days/2026-09-15-sigmoid.py   # render (writes GIFs into site/<slug>/)
python build.py                     # GIF -> MP4, then .py -> HTML
python serve.py                     # preview at :8092, same bytes as published
```

`build.py` output is committed, so Pages serves it directly -- no CI, and the
local preview can never disagree with the published page.
