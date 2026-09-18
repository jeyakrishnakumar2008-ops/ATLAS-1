"""
api/index.py
============
Vercel Serverless Entrypoint for ATLAS Problem 1 + Problem 2 MONITOR.
Provides standard WSGI application (`app`) and BaseHTTPRequestHandler (`handler`).
Pure Python standard library — zero external frameworks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import dispatch_request, AtlasRequestHandler, _extract_routed_path_and_query


def app(environ, start_response):
    """
    WSGI entrypoint for Vercel Python runtime.
    Robustly resolves paths rewritten by vercel.json or forwarded by reverse proxies.
    """
    method = environ.get("REQUEST_METHOD", "GET").upper()
    query_str = environ.get("QUERY_STRING", "")

    # 1. Check if rewrite passed explicit path parameter (?__path=...)
    params = parse_qs(query_str, keep_blank_values=True)
    raw_path = None
    for param_key in ("__path", "__vercel_path", "original_path"):
        if param_key in params:
            raw_path = params.pop(param_key)[0]
            # Reconstruct clean query_str without routing parameter
            query_str = urlencode([(k, v) for k, vals in params.items() for v in vals])
            break

    # 2. Check HTTP proxy headers if not in query params
    if not raw_path:
        for header_name in (
            "HTTP_X_FORWARDED_URI",
            "HTTP_X_ORIGINAL_URI",
            "HTTP_X_VERCEL_FORWARDED_URI",
            "REQUEST_URI",
            "RAW_URI",
        ):
            val = environ.get(header_name, "").strip()
            if val:
                p_clean = val.split("?")[0].strip()
                if p_clean and not p_clean.startswith("/api/index"):
                    raw_path = p_clean
                    break

    # 3. Fallback to PATH_INFO or default to "/"
    if not raw_path:
        p_info = environ.get("PATH_INFO", "/").strip()
        if p_info and not p_info.startswith("/api/index"):
            raw_path = p_info
        else:
            raw_path = "/"

    # 4. Normalize path
    if not raw_path.startswith("/"):
        raw_path = "/" + raw_path
    if raw_path in ("/api/index.py", "/api/index", ""):
        raw_path = "/"

    # 5. Read body
    content_len = int(environ.get("CONTENT_LENGTH", 0) or 0)
    body_bytes = environ["wsgi.input"].read(content_len) if content_len > 0 else b""

    # 6. Execute request with server-side diagnostic logging
    try:
        status_code, resp_headers, resp_body = dispatch_request(
            method=method,
            path=raw_path,
            query_str=query_str,
            body_bytes=body_bytes,
        )
    except Exception as exc:
        import traceback
        sys.stderr.write(f"[ATLAS VERCEL ERROR] Exception in app() handler for {method} {raw_path}: {exc}\n")
        traceback.print_exc(file=sys.stderr)
        status_code = 500
        resp_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
        }
        resp_body = b'{"status": "error", "question_type": "unknown", "answer": "An internal error occurred on Vercel while processing the query.", "record_refs": []}'

    status_phrases = {
        200: "200 OK",
        201: "201 Created",
        204: "204 No Content",
        400: "400 Bad Request",
        404: "404 Not Found",
        405: "405 Method Not Allowed",
        500: "500 Internal Server Error",
    }
    status_str = status_phrases.get(status_code, f"{status_code} Status")

    start_response(status_str, list(resp_headers.items()))
    return [resp_body]


# Fallback class handler if Vercel uses BaseHTTPRequestHandler
class handler(AtlasRequestHandler):
    pass
