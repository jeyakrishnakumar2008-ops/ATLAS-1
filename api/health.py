"""
api/health.py
=============
Dedicated Vercel serverless function for /api/health.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import dispatch_request, AtlasRequestHandler


def app(environ, start_response):
    status_code, resp_headers, resp_body = dispatch_request("GET", "/api/health", "", b"")
    start_response(f"{status_code} OK", list(resp_headers.items()))
    return [resp_body]


class handler(AtlasRequestHandler):
    def do_GET(self) -> None:
        status, headers, body = dispatch_request("GET", "/api/health", "", b"")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
