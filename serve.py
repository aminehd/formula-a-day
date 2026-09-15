"""Serves playground/out over HTTP. Auto-refreshes when a render lands."""
import http.server, socketserver, os
from pathlib import Path

import json, sys, urllib.parse

# Serve any folder: python serve.py [DIR] [PORT]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
      Path(__file__).resolve().parent / "docs"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PORT", 8092))


def listing():
    """Every image in the folder, newest first, enriched from the manifest."""
    meta = {}
    try:
        for r in json.loads((OUT / "manifest.json").read_text()):
            meta[r["gif"]] = r
    except Exception:
        pass
    rows = []
    for f in OUT.iterdir():
        if f.suffix.lower() not in (".gif", ".png", ".jpg", ".webp"):
            continue
        m = meta.get(f.name, {})
        rows.append({"name": m.get("name", f.stem), "gif": f.name,
                     "code": m.get("code", ""), "err": m.get("err"),
                     "secs": m.get("secs"), "kb": f.stat().st_size // 1024,
                     "ts": f.stat().st_mtime})
    rows.sort(key=lambda r: -r["ts"])
    return rows

PAGE = """<!doctype html><meta charset=utf-8><title>jaxvis playground</title>
<style>
:root{--bg:#0b0b10;--fg:#e9e6f0;--dim:#8b86a0;--card:#15151f;--line:#26263a}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:16px 22px;border-bottom:1px solid var(--line);
 display:flex;gap:14px;align-items:baseline;position:sticky;top:0;
 background:var(--bg);z-index:9}
h1{font-size:15px;margin:0;letter-spacing:.06em;text-transform:uppercase}
.dim{color:var(--dim);font-size:12px}
main{padding:22px;display:grid;gap:22px;
 grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 overflow:hidden}
.card img{width:100%;display:block;background:#000}
pre{margin:0;padding:12px 14px;font-size:12px;overflow-x:auto;
 border-top:1px solid var(--line);white-space:pre}
.name{padding:10px 14px;font-weight:600;display:flex;justify-content:space-between}
.err{color:#ff8098;padding:12px 14px;font-size:12px}
.empty{padding:40px 22px;color:var(--dim)}
</style>
<header><h1>jaxvis playground</h1>
<span class=dim id=st>waiting…</span></header>
<main id=m></main>
<script>
let seen="";
async function tick(){
 try{
  const r=await fetch("api.json?"+Date.now());
  const t=await r.text(); if(t===seen) return; seen=t;
  const rows=JSON.parse(t);
  document.getElementById("st").textContent=
    rows.length+" render"+(rows.length==1?"":"s")+" · "+new Date().toLocaleTimeString();
  document.getElementById("m").innerHTML=rows.map(d=>`<div class=card>
   ${d.err?`<div class=err>${d.err}</div>`
          :`<a href="f/${encodeURIComponent(d.gif)}" target=_blank
              ><img src="f/${encodeURIComponent(d.gif)}?${d.ts}" alt="${d.name}"></a>`}
   <div class=name><span>${d.name}</span>
     <span class=dim>${d.kb} KB${d.secs?" · "+d.secs+"s":""}</span></div>
   ${d.code?`<pre>${d.code.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}</pre>`:""}
  </div>`).join("");
 }catch(e){ document.getElementById("m").innerHTML=
   '<div class=empty>Nothing here yet. Decorate a function with @show.</div>'; }
}
tick(); setInterval(tick,1500);
</script>"""

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUT), **k)
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api.json":
            b = json.dumps(listing()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        if path.startswith("/f/"):          # /f/<name> -> the file itself
            self.path = "/" + urllib.parse.unquote(path[3:])
            return super().do_GET()
        if path in ("/", "/index.html"):
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        return super().do_GET()
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *a): pass

socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), H) as s:
    print(f"playground on :{PORT}  serving {OUT}")
    s.serve_forever()
