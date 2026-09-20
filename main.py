"""
Root entrypoint for deployment platforms (Railway, Render, etc.).
Exposes the ASGI app as `app` and supports direct execution.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.api import app  # noqa: F401 — re-exported for ASGI servers

if __name__ == "__main__":
    import uvicorn
    port = int(__import__("os").environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
