"""Offline test for _ensure_weight in main.py.

    python3 face_swap_backend/test_download_logic.py

_ensure_weight is what now puts the 600 MB of V2 weights on the box, and it
runs exactly once per machine, at boot, unattended — the worst possible place
to find out it leaves a half-written .onnx behind. It is lifted out of main.py
here and driven against a local HTTP server, with no GPU, no onnxruntime and
no 600 MB download involved.
"""
import contextlib, http.server, io, os, pathlib, sys, threading, urllib.request, urllib.error

HERE = pathlib.Path(__file__).parent
TMP  = HERE / ".dltest"
SRV  = TMP / "srv"

# ── httpx shim: only the three surfaces _ensure_weight touches ──────────────
class _Resp:
    def __init__(self, r): self._r = r; self.headers = {k.lower(): v for k, v in r.headers.items()}
    def raise_for_status(self): pass
    def iter_bytes(self, n):
        while True:
            c = self._r.read(n)
            if not c: return
            yield c
    def __enter__(self): return self
    def __exit__(self, *a): self._r.close()

class _Httpx:
    @staticmethod
    def Timeout(*a, **k): return None
    @staticmethod
    def stream(method, url, **kw):
        try:
            return _Resp(urllib.request.urlopen(url, timeout=10))
        except urllib.error.HTTPError as e:
            raise IOError(f"HTTP {e.code}")

# ── serve a fixture over real HTTP ──────────────────────────────────────────
SRV.mkdir(parents=True, exist_ok=True)
BLOB = b"MODELWEIGHTS" * 250_000            # 3,000,000 bytes
(SRV / "weight.onnx").write_bytes(BLOB)

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: Quiet(*a, directory=str(SRV), **k))
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}/weight.onnx"

# ── load the real function against the shim ────────────────────────────────
def _extract() -> str:
    """Pull _ensure_weight out of main.py rather than copying it, so the test
    cannot quietly drift from the thing it is testing."""
    src = io.open(HERE / "main.py", encoding="utf-8").read()
    a = src.index("def _ensure_weight(")
    b = src.index("def _load_models() -> None:")
    return src[a:b]


ns = {"httpx": _Httpx, "contextlib": contextlib, "pathlib": pathlib, "os": os,
      "print": print, "open": open, "IOError": IOError}
def load(auto="1"):
    ns["_AUTO_DOWNLOAD"] = auto == "1"
    exec(_extract(), ns)
    return ns["_ensure_weight"]

fails = []
def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond: fails.append(msg)

dest = TMP / "out" / "weight.onnx"
ensure = load()

print("\n1. missing file is fetched")
ok(ensure(dest, URL, len(BLOB), "weight.onnx") is True, "returns True")
ok(dest.exists() and dest.read_bytes() == BLOB, "bytes on disk match the source exactly")
ok(not (dest.parent / "weight.onnx.part").exists(), "no .part left behind")

print("\n2. an already-good file is a no-op")
mtime = dest.stat().st_mtime_ns
ok(ensure(dest, URL, len(BLOB), "weight.onnx") is True, "returns True")
ok(dest.stat().st_mtime_ns == mtime, "file not touched (no needless 600 MB re-download every boot)")

print("\n3. a truncated file from a killed boot is replaced, not loaded")
dest.write_bytes(BLOB[:1000])
ok(ensure(dest, URL, len(BLOB), "weight.onnx") is True, "returns True")
ok(dest.read_bytes() == BLOB, "truncated file was re-downloaded in full")

print("\n4. a short body fails closed")
dest.unlink()
ok(ensure(dest, URL, len(BLOB) + 999, "weight.onnx") is False, "returns False on length mismatch")
ok(not dest.exists(), "no truncated .onnx published for the loader to trip over")
ok(not (dest.parent / "weight.onnx.part").exists(), "no .part left behind")

print("\n5. a 404 fails closed")
ok(ensure(dest, URL + ".missing", len(BLOB), "weight.onnx") is False, "returns False")
ok(not dest.exists(), "nothing written")

print("\n6. LUCY_AUTO_DOWNLOAD=0 skips the fetch")
ensure0 = load(auto="0")
ok(ensure0(dest, URL, len(BLOB), "weight.onnx") is False, "returns False")
ok(not dest.exists(), "nothing downloaded")

print("\n7. with the file present, the opt-out still reports it usable")
dest.write_bytes(BLOB)
ok(ensure0(dest, URL, len(BLOB), "weight.onnx") is True, "existing file is used even with auto-download off")

srv.shutdown()
import shutil; shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{len(fails)} FAILURES" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
