"""
api/graph.py
============
Dedicated Vercel serverless function for /api/graph.
Returns interactive Study Knowledge Graph nodes and edges for a subject.
Pure Python standard library — zero external frameworks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import dispatch_request, AtlasRequestHandler


def app(environ, start_response):
    """
    WSGI entrypoint for /api/graph.
    Always returns application/json, never HTML.
    """
    method = environ.get("REQUEST_METHOD", "GET").upper()
    query_str = environ.get("QUERY_STRING", "")
    content_len = int(environ.get("CONTENT_LENGTH", 0) or 0)
    body_bytes = environ["wsgi.input"].read(content_len) if content_len > 0 else b""

    # Clean query parameters (strip routing markers if any)
    params = parse_qs(query_str, keep_blank_values=True)
    clean_params = [(k, v) for k, vals in params.items() if k not in ("__path", "__vercel_path", "original_path") for v in vals]
    from urllib.parse import urlencode
    clean_query = urlencode(clean_params)

    status_code, resp_headers, resp_body = dispatch_request(
        method=method,
        path="/api/graph",
        query_str=clean_query,
        body_bytes=body_bytes,
    )

    status_str = f"{status_code} {'OK' if status_code == 200 else 'Error'}"
    start_response(status_str, list(resp_headers.items()))
    return [resp_body]


class handler(AtlasRequestHandler):
    """
    BaseHTTPRequestHandler entrypoint for /api/graph on Vercel.
    """
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query, keep_blank_values=True)
        from urllib.parse import urlencode
        clean_params = [(k, v) for k, vals in params.items() if k not in ("__path", "__vercel_path", "original_path") for v in vals]
        clean_query = urlencode(clean_params)

        status, headers, body = dispatch_request("GET", "/api/graph", clean_query, b"")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b""
        status, headers, body = dispatch_request("POST", "/api/graph", parsed.query, post_body)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
