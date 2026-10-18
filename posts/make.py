"""Postcards: one GIF that carries the formula, the code and the clip.

A clip on its own is pretty and says nothing. Posted somewhere, it needs to
answer "what am I looking at" and "how would I do this" without a caption,
because captions get cropped and links get ignored. So glue the three
together: formula on top, code under it, animation beside them.

    python posts/make.py          -> docs/posts/*.gif
"""
import io
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
OUT = DOCS / "posts"
BG = (11, 11, 16)
FG = (233, 230, 240)
DIM = (128, 132, 150)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# Token colour by Pygments type. Read as a prefix match, longest first, so
# Name.Function finds its own entry and Name falls through to the default.
SYNTAX = {
    "Keyword": (255, 121, 198),
    "Keyword.Namespace": (255, 121, 198),
    "Name.Decorator": (255, 184, 108),
    "Name.Function": (130, 220, 255),
    "Name.Builtin": (130, 220, 255),
    "Literal.String": (255, 205, 135),
    "Literal.Number": (189, 160, 255),
    "Operator": (150, 155, 178),
    "Punctuation": (150, 155, 178),
    "Comment": DIM,
}


def spans_of(code):
    """Python source -> one list of (text, colour) per line.

    Pygments hands back a flat token stream with newlines buried inside the
    token text, so the lexing and the line breaking are two separate jobs.
    """
    from pygments import lex
    from pygments.lexers import PythonLexer

    def colour(tok):
        name = str(tok).removeprefix("Token.")
        while name:
            if name in SYNTAX:
                return SYNTAX[name]
            name = name.rpartition(".")[0]
        return FG

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


def frames_of(mp4, every=2, height=None):
    "Decode an mp4 to PIL frames, keeping every nth."
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4),
                        f"{td}/%04d.png"], check=True)
        fs = sorted(Path(td).glob("*.png"))[::every]
        out = []
        for f in fs:
            im = Image.open(f).convert("RGB")
            if height and im.height != height:
                w = round(im.width * height / im.height)
                im = im.resize((w, height), Image.LANCZOS)
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


def panel(w, h, tex, code, title, foot):
    """The still half: title, formula, code, footer.

    The code font is FITTED, not chosen: a decorator line is long, and a
    fixed size either overflows the panel or collides with the footer
    depending on the snippet. Shrink until both fit -- but start large and
    stop early, because legibility on a phone is the whole point.
    """
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    pad = round(w * 0.050)
    y = pad

    f_title = ImageFont.truetype(MONO_B, round(w * 0.038))
    d.text((pad, y), title, font=f_title, fill=FG)
    y += round(w * 0.038 * 1.85)

    fi = formula_img(tex, px=round(w * 0.033))
    if fi.width > w - 2 * pad:                 # never let it run off the edge
        s_ = (w - 2 * pad) / fi.width
        fi = fi.resize((round(fi.width * s_), round(fi.height * s_)),
                       Image.LANCZOS)
    im.paste(fi, (pad, y), fi)
    y += fi.height + round(w * 0.034)

    d.line([(pad, y), (w - pad, y)], fill=(38, 38, 48), width=1)
    y += round(w * 0.030)

    rows = spans_of(code.strip("\n"))
    widest = max(len("".join(t for t, _ in r)) for r in rows)
    foot_px = round(w * 0.0245)
    room_w, room_h = w - 2 * pad, h - round(pad * 1.6) - foot_px - y
    size = round(w * 0.042)
    while size > 11:
        f = ImageFont.truetype(MONO_B, size)
        if (d.textlength("M" * widest, font=f) <= room_w
                and len(rows) * size * 1.5 <= room_h):
            break
        size -= 1
    f_code = ImageFont.truetype(MONO_B, size)
    adv = d.textlength("M", font=f_code)      # monospace: one width for all
    lh = round(size * 1.5)
    for row in rows:
        x = pad
        for text, col in row:
            d.text((x, y), text, font=f_code, fill=col)
            x += adv * len(text)
        y += lh

    f_foot = ImageFont.truetype(MONO, foot_px)
    d.text((pad, h - round(pad * 0.9) - foot_px), foot, font=f_foot, fill=DIM)
    return im


def postcard(mp4, tex, code, title, name, foot="jaxvis", every=2,
             clip_px=600, panel_ratio=1.52, colors=128):
    # Discord rejects anything over 10 MB, so the palette and the pixel
    # height are the two knobs that matter. A dense 60k-point cloud dithers
    # badly and needs a smaller frame than a line drawing does.
    fs = frames_of(mp4, every=every, height=clip_px)
    ch = fs[0].height
    pw = round(ch * panel_ratio)
    still = panel(pw, ch, tex, code, title, foot)
    out = []
    for f in fs:
        c = Image.new("RGB", (pw + f.width, ch), BG)
        c.paste(still, (0, 0))
        c.paste(f, (pw, 0))
        out.append(c.convert("P", palette=Image.ADAPTIVE, colors=colors))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.gif"
    out[0].save(p, save_all=True, append_images=out[1:], loop=0,
                duration=70, optimize=True, disposal=2)
    mb = p.stat().st_size / 1e6
    print(f"  {p.name}  {len(out)} frames  {c.size[0]}x{c.size[1]}  {mb:.1f} MB")
    return p


POSTS = [
    dict(
        name="sigmoid",
        mp4=DOCS / "2026-10-17-one-dimension" / "sigmoid.mp4",
        title="the squash, one primitive at a time",
        tex=r"$\sigma(x)=\dfrac{1}{1+e^{-x}}$",
        code="""import jaxvis, jax.numpy as jnp

x = jaxvis.rail(-6.0, 6.0, 500)

@jaxvis.analyze(x, **jaxvis.curve(x))
def sigmoid(p):
    return 1.0 / (1.0 + jnp.exp(-p))""",
        foot="github.com/aminehd/jaxvis",
    ),
    dict(
        name="cloud",
        mp4=DOCS / "2026-10-15-building-blocks" / "the_squash.mp4",
        title="60,000 points, pressed into a square",
        tex=r"$y=\tanh(kx)$",
        clip_px=460, colors=96,
        code="""import jaxvis, jax.numpy as jnp

g = sheet()        # 60,000 points

@jaxvis.analyze(g, **jaxvis.cloud(g))
def the_squash(p):
    return jnp.tanh(1.6 * p)""",
        foot="github.com/aminehd/jaxvis",
    ),
    dict(
        name="provenance",
        mp4=DOCS / "2026-10-18-came-from" / "permutation.mp4",
        title="colour means WHICH INPUT, not value",
        tex=r"$y=Mx,\ \ M_{ij}=[\,i+j=n-1\,]$",
        code="""import jaxvis, jax.numpy as jnp

i = jnp.arange(8)
M = (i[:, None] + i[None, :] == 7) * 1.0

jaxvis.grains(M, x)""",
        foot="github.com/aminehd/jaxvis",
    ),
]

if __name__ == "__main__":
    for p in POSTS:
        postcard(**p)
