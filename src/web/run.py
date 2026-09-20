"""
Runner script for the Ninety+ web application.
Starts Uvicorn ASGI server and opens the browser.
"""

import sys
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    port = 8000
    print(f"\n=======================================================")
    print(f"Starting Ninety+ Football Intelligence Platform...")
    print(f"Local URL: http://localhost:{port}")
    print(f"=======================================================\n")
    uvicorn.run("src.web.api:app", host="127.0.0.1", port=port, log_level="info")
