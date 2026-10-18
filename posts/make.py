"""Postcards: one GIF that carries the formula, the code and the clip.

A clip on its own is pretty and says nothing. Posted somewhere, it needs to
answer "what am I looking at" and "how would I do this" without a caption,
because captions get cropped and links get ignored. So glue the three
together: formula on top, code under it, animation beside them.

    python posts/make.py          -> docs/posts/*.gif
"""
import io
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
OUT = DOCS / "posts"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# Two themes. Syntax colours are keyed by Pygments token type and read as a
# prefix match, longest first, so Name.Function finds its own entry while a
# bare Name falls through to the body colour.
DARK = dict(
    bg=(11, 11, 16), fg=(233, 230, 240), dim=(128, 132, 150),
    rule=(38, 38, 48), invert=False,
    syntax={
        "Keyword": (255, 121, 198),
        "Keyword.Namespace": (255, 121, 198),
        "Name.Decorator": (255, 184, 108),
        "Name.Function": (130, 220, 255),
        "Name.Builtin": (130, 220, 255),
        "Literal.String": (255, 205, 135),
        "Literal.Number": (189, 160, 255),
        "Operator": (150, 155, 178),
        "Punctuation": (150, 155, 178),
        "Comment": (128, 132, 150),
    },
)
LIGHT = dict(
    bg=(255, 255, 255), fg=(26, 26, 34), dim=(138, 142, 158),
    rule=(222, 222, 230), invert=False,   # the clip stays on black: it is
                                          # drawn as glow, and glow needs it
    syntax={
        "Keyword": (197, 20, 120),
        "Keyword.Namespace": (197, 20, 120),
        "Name.Decorator": (176, 96, 12),
        "Name.Function": (18, 110, 172),
        "Name.Builtin": (18, 110, 172),
        "Literal.String": (160, 88, 20),
        "Literal.Number": (112, 58, 190),
        "Operator": (122, 126, 144),
        "Punctuation": (122, 126, 144),
        "Comment": (150, 154, 168),
    },
)
THEMES = dict(dark=DARK, light=LIGHT)
BG, FG, DIM = DARK["bg"], DARK["fg"], DARK["dim"]


def on_white(im, chroma=1.0, lift=0.0):
    """Glow-on-black -> ink-on-white, keeping the hue.

    A plain RGB inversion turns the background white but swaps every colour
    for its complement (magenta -> green). Rotating the hue 180 degrees back
    afterwards undoes exactly that, and leaves brightness meaning what it
    meant before: a faint glow becomes faint ink, not a bright one.

    PASTEL clips need the two knobs. A pale pink dot on black inverts to a
    near-black one, so the colour -- which in the grains clip IS the
    information -- disappears. `chroma` scales saturation back up and `lift`
    raises the floor on brightness, turning dark ink back into strong ink.
    """
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    a = 1.0 - a
    mx, mn = a.max(2), a.min(2)
    v, c = mx, mx - mn
    s_ = np.where(mx > 0, c / np.maximum(mx, 1e-6), 0.0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.zeros_like(v)
    nz = c > 1e-6
    h = np.where(nz & (mx == r), ((g - b) / np.maximum(c, 1e-6)) % 6.0, h)
    h = np.where(nz & (mx == g), (b - r) / np.maximum(c, 1e-6) + 2.0, h)
    h = np.where(nz & (mx == b), (r - g) / np.maximum(c, 1e-6) + 4.0, h)
    h = (h / 6.0 + 0.5) % 1.0
    s_ = np.clip(s_ * chroma, 0.0, 1.0)
    v = lift + (1.0 - lift) * v
    i = (np.floor(h * 6.0).astype(int)) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    pp, q, t = v * (1 - s_), v * (1 - f * s_), v * (1 - (1 - f) * s_)
    ch = np.stack([np.stack([v, t, pp], -1), np.stack([q, v, pp], -1),
                   np.stack([pp, v, t], -1), np.stack([pp, q, v], -1),
                   np.stack([t, pp, v], -1), np.stack([v, pp, q], -1)])
    out = np.take_along_axis(ch, i[None, ..., None], 0)[0]

    # The clip's background is a dark PURPLE, not black, so the inversion
    # lands on lavender and the panel's pure white shows the seam. White-
    # balance against the border, which is background by construction.
    edge = np.concatenate([out[:2].reshape(-1, 3), out[-2:].reshape(-1, 3),
                           out[:, :2].reshape(-1, 3), out[:, -2:].reshape(-1, 3)])
    ref = np.maximum(np.median(edge, 0), 0.45)
    out = out / ref
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def spans_of(code, th):
    """Python source -> one list of (text, colour) per line.

    Pygments hands back a flat token stream with newlines buried inside the
    token text, so the lexing and the line breaking are two separate jobs.
    """
    from pygments import lex
    from pygments.lexers import PythonLexer

    def colour(tok):
        name = str(tok).removeprefix("Token.")
        while name:
            if name in th["syntax"]:
                return th["syntax"][name]
            name = name.rpartition(".")[0]
        return th["fg"]

    rows, row = [], []
    for tok, text in lex(code, PythonLexer()):
        col = colour(tok)
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i:
                rows.append(row)
                row = []
            if part:
                row.append((part, col))
    rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    return rows


def frames_of(mp4, every=2, width=None):
    "Decode an mp4 to PIL frames, keeping every nth, scaled to one width."
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4),
                        f"{td}/%04d.png"], check=True)
        fs = sorted(Path(td).glob("*.png"))[::every]
        out = []
        for f in fs:
            im = Image.open(f).convert("RGB")
            if width and im.width != width:
                h = round(im.height * width / im.width)
                im = im.resize((width, h), Image.LANCZOS)
            out.append(im)
        return out


def formula_img(tex, px=34, color=FG):
    """LaTeX -> an RGBA image with real transparency.

    math_to_image saves PNG on an opaque white page, which pastes onto a
    dark panel as a white box. Rendering it black-on-white and using the
    INVERSE as the alpha channel gives a clean cut-out, and lets the glyphs
    be any colour afterwards.
    """
    buf = io.BytesIO()
    mathtext.math_to_image(tex, buf, dpi=px * 6, format="png", color="black")
    buf.seek(0)
    g = Image.open(buf).convert("L")
    alpha = Image.eval(g, lambda v: 255 - v)
    im = Image.new("RGBA", g.size, color + (0,))
    im.putalpha(alpha)
    scale = (px * 2.2) / im.height
    return im.resize((max(1, round(im.width * scale)),
                      max(1, round(im.height * scale))), Image.LANCZOS)


def panel(w, tex, code, foot, th):
    """The top row: formula, code, footer. Height follows the content.

    In the stacked layout the width is fixed by the clip underneath, so the
    font is a fraction of it rather than something fitted -- except for the
    one guard that a long line must still not run off the edge.
    """
    pad = round(w * 0.055)
    rows = spans_of(code.strip("\n"), th)
    widest = max(len("".join(t for t, _ in r)) for r in rows)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    size = round(w * 0.0335)
    while size > 9 and probe.textlength("M" * widest,
                                        font=ImageFont.truetype(MONO, size)) > w - 2 * pad:
        size -= 1
    f_code = ImageFont.truetype(MONO, size)
    lh = round(size * 1.55)

    fi = formula_img(tex, px=round(w * 0.036), color=th["fg"])
    if fi.width > w - 2 * pad:                 # never let it run off the edge
        s_ = (w - 2 * pad) / fi.width
        fi = fi.resize((round(fi.width * s_), round(fi.height * s_)),
                       Image.LANCZOS)

    foot_px = round(w * 0.024)
    gap = round(w * 0.032)
    h = pad + fi.height + gap + gap + len(rows) * lh + gap + foot_px + pad

    im = Image.new("RGB", (w, h), th["bg"])
    d = ImageDraw.Draw(im)
    y = pad
    im.paste(fi, (pad, y), fi)
    y += fi.height + gap

    d.line([(pad, y), (w - pad, y)], fill=th["rule"], width=1)
    y += gap

    adv = d.textlength("M", font=f_code)      # monospace: one width for all
    for row in rows:
        x = pad
        for text, col in row:
            d.text((x, y), text, font=f_code, fill=col)
            x += adv * len(text)
        y += lh
    y += gap

    f_foot = ImageFont.truetype(MONO, foot_px)
    d.text((pad, y), foot, font=f_foot, fill=th["dim"])
    return im


def postcard(mp4, tex, code, name, foot="github.com/aminehd/jaxvis", every=2,
             theme="light", width=620, colors=128, white=None, title=None):
    """One column: the code above, the animation below.

    Discord scales an image down to its column width, so a WIDE card makes
    the code tiny no matter how big it was in the file. Stacking keeps the
    card narrow, which is what actually keeps the text readable in a post.
    """
    th = THEMES[theme]
    white = white or {}
    fs = frames_of(mp4, every=every, width=width)
    if th["invert"]:
        fs = [on_white(f, **white) for f in fs]
    still = panel(width, tex, code, foot, th)
    h = still.height + fs[0].height

    out = []
    for f in fs:
        c = Image.new("RGB", (width, h), th["bg"])
        c.paste(still, (0, 0))
        c.paste(f, (0, still.height))
        out.append(c.convert("P", palette=Image.ADAPTIVE, colors=colors))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"{name}.gif" if theme == "light" else f"{name}-{theme}.gif")
    out[0].save(p, save_all=True, append_images=out[1:], loop=0,
                duration=70, optimize=True, disposal=2)
    mb = p.stat().st_size / 1e6
    print(f"  {p.name}  {len(out)} frames  {width}x{h}  {mb:.1f} MB")
    return p


POSTS = [
    dict(
        name="sigmoid",
        mp4=DOCS / "2026-10-17-one-dimension" / "sigmoid.mp4",
        tex=r"$\sigma(x)=\dfrac{1}{1+e^{-x}}$",
        code="""x = jaxvis.rail(-6.0, 6.0, 500)

@jaxvis.analyze(x, **jaxvis.curve(x))
def sigmoid(p):
    return 1.0 / (1.0 + jnp.exp(-p))""",
    ),
    dict(
        name="tanh",
        mp4=DOCS / "2026-10-17-one-dimension" / "tanh.mp4",
        tex=r"$\tanh(x)=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$",
        every=3,
        code="""@jaxvis.analyze(x, **jaxvis.curve(x))
def tanh(p):
    up, down = jnp.exp(p), jnp.exp(-p)
    return (up - down) / (up + down)""",
    ),
    dict(
        name="gelu",
        mp4=DOCS / "2026-10-17-one-dimension" / "gelu.mp4",
        tex=r"$\mathrm{gelu}(x)=\dfrac{x}{2}\left(1+\tanh\left[c\,(x+0.045x^{3})\right]\right)$",
        every=4,
        code="""@jaxvis.analyze(x, **jaxvis.curve(x))
def gelu(p):
    inner = 0.798 * (p + 0.045 * p ** 3)
    return 0.5 * p * (1 + jnp.tanh(inner))""",
    ),
    dict(
        name="cloud",
        mp4=DOCS / "2026-10-15-building-blocks" / "the_squash.mp4",
        tex=r"$y=\tanh(kx)$",
        width=460, colors=48, every=3,
        code="""g = sheet()     # 60,000 points

@jaxvis.analyze(g, **jaxvis.cloud(g))
def the_squash(p):
    return jnp.tanh(1.6 * p)""",
    ),
    dict(
        name="provenance",
        mp4=DOCS / "2026-10-18-came-from" / "permutation.mp4",
        tex=r"$y=Mx,\ \ M_{ij}=[\,i+j=n-1\,]$",
        white=dict(chroma=2.4, lift=0.30),
        code="""i = jnp.arange(8)
M = (i[:, None] + i[None, :] == 7) * 1.0

jaxvis.grains(M, x)""",
    ),
]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    for p in POSTS:
        if not only or p["name"] in only:
            postcard(**p)
