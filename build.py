"""days/*.py -> docs/. Run after rendering; converts GIFs and writes HTML.

The day file IS the page: `# %% [markdown]` blocks become prose, code blocks
become highlighted code, and any GIF the file rendered is embedded as MP4.
Local output is byte-identical to what GitHub Pages serves -- there is no
separate build step on the server, so preview never disagrees with published.
"""
import re
import sys
from pathlib import Path

import markdown as md
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent
DAYS, SITE = ROOT / "days", ROOT / "docs"
FMT = HtmlFormatter(style="monokai", nowrap=False)


def cells(text):
    """Percent-format -> [('md'|'code', body)]."""
    out, kind, buf = [], "code", []
    for line in text.splitlines():
        m = re.match(r"^#\s*%%(.*)$", line)
        if m:
            if buf:
                out.append((kind, "\n".join(buf).strip("\n")))
            kind = "md" if "markdown" in m.group(1) else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        out.append((kind, "\n".join(buf).strip("\n")))
    # markdown cells are comment blocks -- strip the leading "# "
    return [(k, re.sub(r"(?m)^# ?", "", b) if k == "md" else b)
            for k, b in out if b.strip()]


def to_mp4(d):
    from jaxvis.publish import to_mp4 as conv
    made = []
    for g in sorted(d.glob("*.gif")):
        try:
            p, gk, mk = conv(g)
            made.append((p.name, gk, mk))
            g.unlink()                      # the MP4 replaces it
        except Exception as e:
            print(f"  ! {g.name}: {e}")
    return made


HEAD = """<!doctype html><meta charset=utf-8><title>{title}</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#0b0b10;--fg:#e9e6f0;--dim:#8b86a0;--line:#26263a;--card:#15151f}}
body{{margin:0 auto;max-width:820px;padding:30px 20px 90px;background:var(--bg);
 color:var(--fg);font:17px/1.75 -apple-system,Segoe UI,Roboto,sans-serif}}
a{{color:#8ad7ff}} h1{{font-size:30px}} h2{{margin-top:38px;font-size:22px}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:14px}}
p code{{background:var(--card);padding:1px 5px;border-radius:4px}}
.highlight{{background:var(--card);border:1px solid var(--line);
 border-radius:8px;padding:12px 14px;overflow-x:auto;margin:16px 0}}
video{{width:52%;border-radius:8px;margin:14px 0;background:#000;display:block}}
@media(max-width:620px){{video{{width:100%}}}}
.back{{color:var(--dim);text-decoration:none;font-size:13px}}
.dim{{color:var(--dim);font-size:14px}}
.gal{{font-size:14px;color:var(--dim)}}
.gal video{{width:100%;margin:6px 0 2px}}
.gal h3{{font-size:16px;margin:26px 0 2px;color:var(--fg)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:620px){{.grid2{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse}} td,th{{border:1px solid var(--line);padding:5px 9px}}
{pyg}
</style>
"""


def day_body(f, vids, prefix=""):
    """The day's content as HTML chunks. Shared by its own page and the index,
    so the long scroll and the permalink can never drift apart."""
    out = []
    for kind, body in cells(f.read_text()):
        if kind == "md":
            out.append(md.markdown(body, extensions=["tables", "fenced_code"]))
        else:
            out.append(highlight(body, PythonLexer(), FMT))
            for fn in re.findall(r"^def (\w+)", body, re.M):
                if f"{fn}.mp4" in vids:
                    out.append(f'<video src="{prefix}{fn}.mp4" autoplay loop '
                               f'muted playsinline></video>')
    return out


def render_day(f):
    slug = f.stem
    d = SITE / slug
    d.mkdir(parents=True, exist_ok=True)
    made = to_mp4(d)
    title = slug.split("-", 3)[-1].replace("-", " ")
    vids = sorted(p.name for p in d.glob("*.mp4"))
    html = [HEAD.format(title=title, pyg=FMT.get_style_defs(".highlight")),
            '<a class=back href="../index.html">&larr; all formulas</a>']
    html += day_body(f, vids)
    (d / "index.html").write_text("\n".join(html))
    return slug, title, vids, made


def main():
    SITE.mkdir(exist_ok=True)
    days, rows = [], []
    for f in sorted(DAYS.glob("*.py"), reverse=True):
        slug, title, vids, made = render_day(f)
        for n, gk, mk in made:
            print(f"  {n}  {gk} KB -> {mk} KB")
        days.append((f, slug, title, vids))

    idx = [HEAD.format(title="formula a day",
                       pyg=FMT.get_style_defs(".highlight"))]
    intro = ROOT / "intro.md"
    if intro.exists():
        idx.append(md.markdown(intro.read_text(),
                               extensions=["tables", "fenced_code"]))
    if len(days) > 1:                       # a jump list, once there are a few
        idx.append("<p class=dim>" + " &middot; ".join(
            f'<a href="#{s}">{ti}</a>' for _, s, ti, _ in days) + "</p>")
    for f, slug, title, vids in days:
        idx.append(f'<hr id="{slug}" style="border:0;border-top:1px solid '
                   f'var(--line);margin:46px 0 22px">')
        idx.append(f'<div class=dim>{slug.rsplit("-", 1)[0][:10]} &middot; '
                   f'<a class=back href="{slug}/index.html">permalink</a></div>')
        idx += day_body(f, vids, prefix=f"{slug}/")
    gal = ROOT / "gallery.md"
    gdir = SITE / "gallery"
    if gal.exists() and gdir.exists():
        idx.append('<hr style="border:0;border-top:1px solid var(--line);'
                   'margin:52px 0 10px">')
        body = md.markdown(gal.read_text(), extensions=["tables", "fenced_code"])
        # every "- **name** -- ..." bullet becomes a captioned clip
        names = re.findall(r"<strong>(\w+)</strong>", body)
        idx.append(body.split("<ul>")[0])
        cards = []
        for n in names:
            if not (gdir / f"{n}.mp4").exists():
                continue
            cap = re.search(rf"<strong>{n}</strong>\s*(?:&[a-z]+;|-|\u2014)*\s*"
                            rf"(.*?)</li>", body, re.S)
            txt = re.sub(r"<[^>]+>", "", cap.group(1)).strip() if cap else ""
            cards.append(f'<div><h3>{n}</h3>'
                         f'<video src="gallery/{n}.mp4" autoplay loop muted '
                         f'playsinline></video><div>{txt}</div></div>')
        idx.append(f'<div class="gal grid2">{"".join(cards)}</div>')
    (SITE / "index.html").write_text("\n".join(idx))
    print(f"  built {len(days)} day(s) -> docs/  (index is the full scroll)")


if __name__ == "__main__":
    sys.exit(main())
