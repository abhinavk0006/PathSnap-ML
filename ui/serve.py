"""Serve the evaluator UI (Laptop 2):  python ui/serve.py [--port 8080] [--api http://<laptop1-ip>:8000]"""
import argparse
import http.server
import socketserver
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--api", default="", help="API base URL, e.g. http://192.168.1.20:8000 (written into config.js)")
a = ap.parse_args()
root = Path(__file__).resolve().parent
if a.api:
    (root / "config.js").write_text('window.PATHSNAP_CONFIG = { apiUrl: "%s" };\n' % a.api.rstrip("/"))


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):
        super().__init__(*args, directory=str(root), **kw)


with socketserver.TCPServer(("0.0.0.0", a.port), H) as s:
    print(f"PathSnap UI on http://0.0.0.0:{a.port}  (API: {a.api or 'set in the UI header'})")
    s.serve_forever()
