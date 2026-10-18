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
    depending on the snippet. Shrink until both fit.
    """
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    pad = round(w * 0.065)
    y = pad

    f_title = ImageFont.truetype(MONO, round(w * 0.038))
    d.text((pad, y), title, font=f_title, fill=FG)
    y += round(w * 0.038 * 2.2)

    fi = formula_img(tex, px=round(w * 0.032))
    if fi.width > w - 2 * pad:                 # never let it run off the edge
        s_ = (w - 2 * pad) / fi.width
        fi = fi.resize((round(fi.width * s_), round(fi.height * s_)),
                       Image.LANCZOS)
    im.paste(fi, (pad, y), fi)
    y += fi.height + round(w * 0.05)

    d.line([(pad, y), (w - pad, y)], fill=(38, 38, 48), width=1)
    y += round(w * 0.042)

    lines = code.strip("\n").split("\n")
    foot_px = round(w * 0.0245)
    room_w, room_h = w - 2 * pad, h - pad * 2 - foot_px - y
    size = round(w * 0.030)
    while size > 9:
        f = ImageFont.truetype(MONO, size)
        wide = max(d.textlength(l, font=f) for l in lines)
        if wide <= room_w and len(lines) * size * 1.62 <= room_h:
            break
        size -= 1
    f_code = ImageFont.truetype(MONO, size)
    lh = round(size * 1.62)
    for line in lines:
        col = DIM if line.lstrip().startswith("#") else FG
        d.text((pad, y), line, font=f_code, fill=col)
        y += lh

    f_foot = ImageFont.truetype(MONO, foot_px)
    d.text((pad, h - pad - foot_px), foot, font=f_foot, fill=DIM)
    return im


def postcard(mp4, tex, code, title, name, foot="jaxvis", every=2,
             clip_px=600, panel_ratio=1.32, colors=128):
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

@jaxvis.analyze(x, **jaxvis.curve(x, ylim=(-0.35, 1.35)))
def sigmoid(p):
    return 1.0 / (1.0 + jnp.exp(-p))

# every frame is a real intermediate value,
# pulled straight out of the jaxpr""",
        foot="github.com/aminehd/jaxvis",
    ),
    dict(
        name="cloud",
        mp4=DOCS / "2026-10-15-building-blocks" / "the_squash.mp4",
        title="60,000 points, pressed into a square",
        tex=r"$y=\tanh(kx)$",
        clip_px=460, colors=96,
        code="""import jaxvis, jax.numpy as jnp
from jaxvis import tint

g = sheet()            # 60,000 random points

@jaxvis.analyze(g, colors=tint.by_radius(g),
                **jaxvis.cloud(g, palette="bloom"))
def the_squash(p):
    return jnp.tanh(1.6 * p)

# the saturation every activation is buying""",
        foot="github.com/aminehd/jaxvis",
    ),
    dict(
        name="provenance",
        mp4=DOCS / "2026-10-18-came-from" / "permutation.mp4",
        title="colour means WHICH INPUT, not value",
        tex=r"$y=Mx,\ \ M_{ij}=[\,i+j=n-1\,]$",
        code="""import jaxvis, jax.numpy as jnp

i = jnp.arange(8)

jaxvis.grains((i[:, None] + i[None, :] == 7) * 1.0, x)

# each input is a 9x9 block of coloured grains,
# split among the outputs by how much each used.
# a permutation gives you eight pure colours.""",
        foot="github.com/aminehd/jaxvis",
    ),
]

if __name__ == "__main__":
    for p in POSTS:
        postcard(**p)
