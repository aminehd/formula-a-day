"""Local dev server for formula-a-day.

Serves docs/ -- the exact bytes GitHub Pages serves -- and watches days/*.py,
intro.md and gallery.md. Touch any of them and it re-renders and rebuilds, then
the open page reloads itself. Edit a number, glance at the browser.

    python serve.py [PORT]

The reload script is injected ON THE WAY OUT, never written into docs/, so the
published site stays clean.
"""
import hashlib
import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "docs"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8092))
PY_ = sys.executable

VERSION = "0"
STATUS = "ready"
ERROR = ""


def _watched():
    fs = sorted((ROOT / "days").glob("*.py"))
    fs += [p for p in (ROOT / "intro.md", ROOT / "gallery.md") if p.exists()]
    return {p: p.stat().st_mtime for p in fs}


def _run(cmd, label):
    global STATUS, ERROR
    STATUS = label
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        # Last meaningful line of the traceback -- usually the SyntaxError or
        # the exception, which is what you actually need to see.
        tail = [l for l in r.stderr.strip().splitlines() if l.strip()]
        ERROR = tail[-1][:160] if tail else f"{label} failed"
        print(f"  ! {label}\n{r.stderr[-1200:]}")
    return r.returncode == 0


def rebuild(days):
    """Re-render only the day files that changed, then rebuild the site."""
    global VERSION, STATUS, ERROR
    ERROR = ""
    for d in days:
        print(f"  rendering {d.name} ...")
        _run([PY_, str(d)], f"rendering {d.name}")
    print("  building ...")
    _run([PY_, str(ROOT / "build.py")], "building")
    h = hashlib.md5()
    for f in sorted(SITE.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(str(f.stat().st_mtime_ns).encode())
    VERSION = h.hexdigest()[:12]
    STATUS = "ready"
    print(f"  ready ({VERSION})")


def watcher():
    prev = _watched()
    rebuild([])                       # version stamp for whatever is on disk
    while True:
        time.sleep(1.0)
        try:
            now = _watched()
        except OSError:
            continue
        changed = [p for p, m in now.items() if prev.get(p) != m]
        if changed:
            prev = now
            rebuild([p for p in changed if p.suffix == ".py"])


RELOAD = """
<div id=_st style="position:fixed;right:12px;bottom:12px;z-index:99;
 font:12px ui-monospace,monospace;background:#15151f;color:#8b86a0;
 border:1px solid #26263a;border-radius:6px;padding:5px 9px">live</div>
<script>
let _v=null;
setInterval(async()=>{try{
  const r=await fetch('/__v?'+Date.now());
  const [v,s,err]=(await r.text()).split('\x1f');
  const el=document.getElementById('_st');
  if(err){ el.textContent='\u26a0 '+err; el.style.color='#ff8098';
           el.style.maxWidth='70vw'; el.style.whiteSpace='normal'; }
  else { el.textContent = s==='ready' ? 'live' : s; el.style.color='#8b86a0'; }
  if(_v===null){_v=v;} else if(v!==_v){location.reload();}
}catch(e){}}, 900);
</script>
"""


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(SITE), **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/__v":
            return self._send(f"{VERSION}\x1f{STATUS}\x1f{ERROR}".encode(),
                              "text/plain")
        p = self.path.split("?")[0]
        f = SITE / (p.lstrip("/") or "index.html")
        if f.is_dir():
            f = f / "index.html"
        if f.suffix == ".html" and f.is_file():
            return self._send(f.read_bytes() + RELOAD.encode(), "text/html")
        return super().do_GET()

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


threading.Thread(target=watcher, daemon=True).start()
socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), H) as s:
    print(f"formula-a-day dev server on :{PORT}  watching days/ intro.md gallery.md")
    s.serve_forever()
