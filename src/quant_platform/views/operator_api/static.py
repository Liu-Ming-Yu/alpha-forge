"""Serve the built operator-console SPA from the operator API process.

The console is a static Vite/React bundle that lives outside ``src/`` (under
``<repo>/ui``) and is built to ``<repo>/ui/dist``.  We serve it under the
public ``/app`` prefix so it never collides with an API path and so the auth
middleware can carve it out with a single prefix check (the SPA shell must load
*before* the operator supplies a key; protected JSON endpoints stay gated).

Design notes:

* ``GET /``           → redirect to ``/app/`` (friendly default landing).
* ``GET /app``        → redirect to ``/app/`` (normalise trailing slash).
* ``GET /app/{path}`` → the requested file when it exists and is inside
  ``dist`` (path-traversal is rejected), otherwise ``index.html`` so the
  client-side router can resolve deep links.

When ``dist`` is absent we serve a small build-instructions page instead of
404ing, so a fresh checkout still renders something useful at ``/app``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.config import PlatformSettings

# static.py → operator_api → views → quant_platform → src → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Hashed Vite assets are content-addressed, so they are safe to cache hard.
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_NO_CACHE = "no-cache"

_PLACEHOLDER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Quant Console — not built</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; min-height:100vh; display:grid; place-items:center;
         background:#000; color:#f5f5f7;
         font:16px/1.6 -apple-system,BlinkMacSystemFont,"SF Pro Text",Inter,sans-serif; }
  .card { max-width:34rem; padding:2.5rem; border-radius:20px;
          background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
          box-shadow:0 20px 60px rgba(0,0,0,.5); }
  h1 { font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.02em; }
  p  { color:#a1a1a6; margin:.5rem 0; }
  code { background:rgba(255,255,255,.08); padding:.15rem .45rem; border-radius:6px;
         font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:.85em; }
  pre { background:rgba(255,255,255,.06); padding:1rem 1.25rem; border-radius:12px;
        overflow:auto; border:1px solid rgba(255,255,255,.06); }
  .dot { color:#0a84ff; }
</style>
</head>
<body>
  <div class="card">
    <h1><span class="dot">●</span> Operator console not built yet</h1>
    <p>The API is running, but the console bundle was not found at
       <code>__DIST_PATH__</code>.</p>
    <p>Build it once with Node:</p>
    <pre>cd ui
npm install
npm run build</pre>
    <p>Then reload this page. For live development run <code>npm run dev</code>
       and open the Vite dev server instead.</p>
  </div>
</body>
</html>
"""


def resolve_console_dist(settings: PlatformSettings) -> Path:
    """Return the configured console ``dist`` directory (may not exist)."""
    configured = settings.api.console_dist_dir.strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_REPO_ROOT / "ui" / "dist").resolve()


def _file_response(path: Path, *, immutable: bool) -> FileResponse:
    headers = {"Cache-Control": _IMMUTABLE_CACHE if immutable else _NO_CACHE}
    return FileResponse(path, headers=headers)


def mount_operator_console(app: FastAPI, settings: PlatformSettings) -> None:
    """Register the public SPA routes on ``app``.

    Routes are intentionally registered *after* the JSON API routes so an API
    path always wins; the SPA only ever owns ``/`` and ``/app**``.
    """
    dist = resolve_console_dist(settings)
    index_html = dist / "index.html"

    @app.get("/", include_in_schema=False)
    async def _console_root() -> RedirectResponse:
        return RedirectResponse(url="/app/", status_code=307)

    @app.get("/app", include_in_schema=False)
    async def _console_app_root() -> RedirectResponse:
        return RedirectResponse(url="/app/", status_code=307)

    @app.get("/app/{full_path:path}", include_in_schema=False, response_model=None)
    async def _console_spa(full_path: str) -> FileResponse | HTMLResponse:
        if not index_html.is_file():
            return HTMLResponse(
                _PLACEHOLDER.replace("__DIST_PATH__", str(dist)),
                status_code=200,
            )
        # Resolve and confine the request to ``dist`` (reject traversal).
        if full_path:
            candidate = (dist / full_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(dist):
                # Anything under /app/assets/* is a hashed, immutable bundle.
                immutable = "assets/" in full_path or full_path.startswith("assets")
                return _file_response(candidate, immutable=immutable)
        # Deep link / unknown path → let the client-side router handle it.
        return _file_response(index_html, immutable=False)
