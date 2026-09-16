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
FMT = HtmlFormatter(style="friendly", nowrap=False)


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
<link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link rel=stylesheet href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[
  {{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}})"></script>
<link rel=stylesheet href="https://fonts.googleapis.com/css2?family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
/* HackerRank-ish: white page, slate text, green accent, dark code. */
:root{{--bg:#ffffff;--fg:#39424e;--dim:#6b7f92;--line:#e4e9f0;--card:#f7f9fb;--accent:#1ba94c}}
body{{margin:0 auto;max-width:1040px;padding:30px 22px 70px;background:var(--bg);
 color:var(--fg);
 /* Typewriter: one monospace face for everything, set small and tight.
    A single width for prose, headings and code makes the page read like a
    printed listing rather than a magazine. */
 font:400 13.5px/1.62 'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
 font-optical-sizing:auto;letter-spacing:-.003em;text-rendering:optimizeLegibility;
 -webkit-font-smoothing:antialiased}}
/* 45-90 characters is the readable range; 980px of 17.5px text is ~115, so the
   PROSE is capped separately and only the grids use the full width. */
p,h1,h2,h3,ul,ol,blockquote{{max-width:74ch}}
.pair{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);
 gap:20px;align-items:start;margin:20px 0;max-width:none}}
.pair .code{{margin:0;max-width:none}}
.pair video{{width:100%;margin:0}}
@media(max-width:780px){{.pair{{grid-template-columns:1fr}}}}
a{{color:var(--accent);text-underline-offset:3px;text-decoration-thickness:1px}}
a:hover{{color:#15843c}}
.katex-display{{margin:.6em 0 1em;font-size:1.02em}}
.dayhead{{font-size:14px;font-weight:600;letter-spacing:.12em;
 text-transform:uppercase;margin:.1em 0 .9em;color:#1d2429}}
h1{{font-size:17px;font-weight:600;letter-spacing:.14em;line-height:1.35;
 margin:0 0 1.1em;color:#1d2429;text-transform:uppercase}}
h2{{color:#2c3540}}
h3{{color:#2c3540}}
h2{{margin-top:30px;margin-bottom:.5em;font-size:12.5px;font-weight:600;
 letter-spacing:.11em;text-transform:uppercase;line-height:1.4;
 color:#54606b}}
p{{margin:0 0 .7em}}
ul,ol{{margin:.4em 0 .9em}}
li{{margin:.15em 0}}
code,pre,.highlight{{font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-ligatures:none}}
p code,li code{{background:var(--card);color:#2f6f4f;padding:2px 6px;border:1px solid var(--line);border-radius:3px;font-size:13px}}
/* Bare. No card, no bar, no shadow, no fill -- hairline rules above and
   below, the way a listing sits in a textbook. Colour is nearly absent: the
   syntax carries weight and italics instead. */
.code{{margin:14px 0;max-width:80ch}}
.code-bar{{display:none}}
.highlight{{background:none;border:0;border-top:1px solid #e8ecf1;
 border-bottom:1px solid #e8ecf1;border-radius:0;padding:16px 0 16px 2px;
 margin:0;font-size:12.5px;line-height:1.66;letter-spacing:0;
 color:#30373f}}
/* no sideways scrolling: long lines wrap, with the continuation indented so
   you can see it is a continuation */
/* No hanging indent: inside a <pre> the whole block is one box, so
   text-indent hits only the first line while padding-left shifts them all --
   which reads as every line but the first being tabbed in. */
.highlight pre{{white-space:pre-wrap;word-break:break-word;margin:0}}
/* neon: highly saturated hues on white. Pale pastels read as washed out at
   12.5px, so these are pushed to full chroma and kept dark enough to pass
   contrast on a white ground. */
.highlight .k,.highlight .kn,.highlight .kc,.highlight .ow{{color:#e6007a;
 font-weight:600}}
.highlight .nb,.highlight .bp{{color:#00a3cc}}
.highlight .nf,.highlight .fm{{color:#7c1fff;font-weight:600}}
.highlight .s,.highlight .s1,.highlight .s2,.highlight .sa{{color:#00994d}}
.highlight .mi,.highlight .mf,.highlight .m{{color:#ff6a00}}
.highlight .o,.highlight .p{{color:#5b6673}}
.highlight .n{{color:#22272e}}
.highlight .c,.highlight .c1,.highlight .cm{{color:#8fa0b0;font-style:normal}}
.highlight .nd{{color:#c400a8;font-weight:600}}
.highlight pre{{margin:0;background:none}}
.highlight .c,.highlight .c1,.highlight .cm{{font-style:normal;opacity:.72}}
video{{width:52%;border-radius:6px;margin:14px 0;background:#0b0b10;display:block;border:1px solid var(--line)}}
@media(max-width:620px){{video{{width:100%}}}}
.back{{color:var(--dim);text-decoration:none;font-size:13px}}
.dim{{color:var(--dim);font-size:11.5px;letter-spacing:.05em}}
.gal h3{{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
 font-weight:600;margin:14px 0 2px}}
.gal{{font-size:14px;color:var(--dim)}}
.gal video{{width:100%;margin:6px 0 2px}}
.gal h3{{font-size:16px;margin:26px 0 2px;color:var(--fg)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:22px}}
.grid4 h3{{font-size:13px;margin:16px 0 2px}}
@media(max-width:860px){{.grid4{{grid-template-columns:1fr 1fr}}}}
@media(max-width:620px){{.grid2,.grid4{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse}} td,th{{border:1px solid var(--line);padding:5px 9px}}
{pyg}
</style>
<script>
addEventListener('click', e => {{
  const b = e.target.closest('.code-bar button'); if (!b) return;
  const code = b.closest('.code').querySelector('pre').innerText;
  navigator.clipboard.writeText(code).then(() => {{
    b.textContent = 'copied'; b.classList.add('ok');
    setTimeout(() => {{ b.textContent = 'copy'; b.classList.remove('ok'); }}, 1400);
  }});
}});
</script>
"""


def stamp(name, sub=None):
    "Modification time of a video, used to bust the browser cache."
    d = SITE / sub if sub else _CUR[0]
    f = d / f"{name}.mp4"
    return int(f.stat().st_mtime) if f.exists() else 0


_CUR = [SITE]


def day_body(f, vids, prefix="", day_dir=None):
    """The day's content as HTML chunks. Shared by its own page and the index,
    so the long scroll and the permalink can never drift apart."""
    _CUR[0] = day_dir or SITE
    out = []
    for kind, body in cells(f.read_text()):
        if kind == "md":
            out.append(md.markdown(body, extensions=["tables", "fenced_code"]))
        else:
            code = ('<div class=code>'
                    + highlight(body, PythonLexer(), FMT) + '</div>')
            # ?v=<mtime>: the page reloads but the browser reuses a cached
            # video when the URL is unchanged. Stamping it makes every
            # re-render a new URL.
            clips = [f'<video src="{prefix}{fn}.mp4?v={stamp(fn)}" autoplay '
                     f'loop muted playsinline></video>'
                     for fn in re.findall(r"^def (\w+)", body, re.M)
                     if f"{fn}.mp4" in vids]
            # a cell that renders something shows the code and the clip as one
            # row; a cell that renders nothing is just code
            out.append(f'<div class=pair>{code}{"".join(clips)}</div>'
                       if clips else code)
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
    html += day_body(f, vids, day_dir=d)
    (d / "index.html").write_text("\n".join(html))
    return slug, title, vids, made


def gallery_html():
    """gallery.md -> captioned clips. Bullets of the form

        - **name** -- caption

    become cards, matched to docs/gallery/<name>.mp4. Any prose before or
    after the bullets is kept, so the file stays editable as plain markdown.
    """
    gal, gdir = ROOT / "gallery.md", SITE / "gallery"
    if not (gal.exists() and gdir.exists()):
        return []
    body = md.markdown(gal.read_text(), extensions=["tables", "fenced_code"])
    names = re.findall(r"<strong>(\w+)</strong>", body)
    head, tail = body.split("<ul>")[0], body.rsplit("</ul>", 1)[-1]
    cards = []
    for n in names:
        if not (gdir / f"{n}.mp4").exists():
            continue
        cap = re.search(rf"<strong>{n}</strong>\s*(?:&[a-z]+;|-|\u2014)*\s*"
                        rf"(.*?)</li>", body, re.S)
        txt = re.sub(r"<[^>]+>", "", cap.group(1)).strip() if cap else ""
        cards.append(f'<div><h3>{n}</h3>'
                     f'<video src="gallery/{n}.mp4?v={stamp(n, "gallery")}" '
                     f'autoplay loop muted '
                     f'playsinline></video><div>{txt}</div></div>')
    lead = f'<div class="gal grid2">{"".join(cards[:2])}</div>'
    rest = (f'<div class="gal grid4">{"".join(cards[2:])}</div>'
            if len(cards) > 2 else "")
    return [head, lead, rest, tail,
            '<hr style="border:0;border-top:1px solid var(--line);'
            'margin:56px 0 10px">']


def main():
    # One build at a time. The dev server rebuilds on every save, so a manual
    # `python build.py` can land mid-convert -- and to_mp4 DELETES the GIF
    # after converting it, so the loser of the race finds nothing there.
    import fcntl
    SITE.mkdir(exist_ok=True)
    lock = open(ROOT / ".build.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
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
    idx += gallery_html()
    if len(days) > 1:                       # a jump list, once there are a few
        idx.append("<p class=dim>" + " &middot; ".join(
            f'<a href="#{s}">{ti}</a>' for _, s, ti, _ in days) + "</p>")
    for f, slug, title, vids in days:
        idx.append(f'<hr id="{slug}" style="border:0;border-top:1px solid '
                   f'var(--line);margin:46px 0 22px">')
        idx.append(f'<div class=dim>{slug.rsplit("-", 1)[0][:10]} &middot; '
                   f'<a class=back href="{slug}/index.html">permalink</a></div>')
        body = day_body(f, vids, prefix=f"{slug}/", day_dir=SITE / slug)
        idx += [c.replace("<h1>", '<div class=dayhead>')
                 .replace("</h1>", "</div>") for c in body]
    (SITE / "index.html").write_text("\n".join(idx))
    print(f"  built {len(days)} day(s) -> docs/  (index is the full scroll)")


if __name__ == "__main__":
    sys.exit(main())
