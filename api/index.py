import os
import sys
import io
import json
import traceback
import http.server
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

IMPORT_ERROR = None
IMPORT_TRACEBACK = None
UniversalRequestHandler = None

try:
    from app.satellite_enhancer import UniversalRequestHandler as _URH
    UniversalRequestHandler = _URH
except Exception as _e:
    IMPORT_ERROR = str(_e)
    IMPORT_TRACEBACK = traceback.format_exc()


class CaseInsensitiveDict(dict):
    """Case-insensitive dictionary wrapper for WSGI request headers."""
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default

    def __getitem__(self, key):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        raise KeyError(key)


def get_handler_class():
    global UniversalRequestHandler, IMPORT_ERROR, IMPORT_TRACEBACK
    if UniversalRequestHandler is None:
        try:
            from app.satellite_enhancer import UniversalRequestHandler as _URH
            UniversalRequestHandler = _URH
            IMPORT_ERROR = None
            IMPORT_TRACEBACK = None
        except Exception as _e:
            IMPORT_ERROR = str(_e)
            IMPORT_TRACEBACK = traceback.format_exc()
    return UniversalRequestHandler


def wsgi_dispatch(environ, start_response):
    """Bridge WSGI invocation to canonical TERRA-SR UniversalRequestHandler routing."""
    handler_cls = get_handler_class()
    if handler_cls is None:
        err_payload = json.dumps({
            "status": "IMPORT_ERROR",
            "error": IMPORT_ERROR,
            "traceback": IMPORT_TRACEBACK
        }, indent=2).encode("utf-8")
        start_response("500 Internal Server Error", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(err_payload))),
            ("Access-Control-Allow-Origin", "*")
        ])
        return [err_payload]

    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    query = environ.get("QUERY_STRING", "")
    full_path = f"{path}?{query}" if query else path

    headers_dict = CaseInsensitiveDict()
    for k, v in environ.items():
        if k.startswith("HTTP_"):
            header_name = k[5:].replace("_", "-")
            headers_dict[header_name] = v
        elif k in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            header_name = k.replace("_", "-")
            headers_dict[header_name] = v

    try:
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
    except (ValueError, TypeError):
        content_length = 0

    input_stream = environ.get("wsgi.input")
    if input_stream is not None and content_length > 0:
        body_bytes = input_stream.read(content_length)
        rfile = io.BytesIO(body_bytes)
    else:
        rfile = io.BytesIO(b"")

    output_buffer = io.BytesIO()
    response_headers = []
    response_status = [200, "OK"]

    handler_instance = object.__new__(handler_cls)
    handler_instance.command = method
    handler_instance.path = full_path
    handler_instance.request_version = "HTTP/1.1"
    handler_instance.headers = headers_dict
    handler_instance.rfile = rfile
    handler_instance.wfile = output_buffer
    handler_instance.close_connection = True

    def send_response(status_code, message=None):
        status_map = {
            200: "OK", 201: "Created", 204: "No Content",
            400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
            404: "Not Found", 405: "Method Not Allowed",
            500: "Internal Server Error"
        }
        response_status[0] = status_code
        response_status[1] = message or status_map.get(status_code, "OK")

    def send_header(keyword, value):
        response_headers.append((str(keyword), str(value)))

    def end_headers():
        pass

    def send_bytes_response(data: bytes, content_type: str, extra_headers: dict = None, status_code: int = 200):
        send_response(status_code)
        send_header("Content-Type", content_type)
        send_header("Content-Length", str(len(data)))
        send_header("Access-Control-Allow-Origin", "*")
        send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        if extra_headers:
            for k, v in extra_headers.items():
                send_header(k, v)
        end_headers()
        output_buffer.write(data)

    handler_instance.send_response = send_response
    handler_instance.send_header = send_header
    handler_instance.end_headers = end_headers
    handler_instance.send_bytes_response = send_bytes_response

    try:
        if method == "GET":
            handler_instance.do_GET()
        elif method == "POST":
            handler_instance.do_POST()
        elif method == "OPTIONS":
            if hasattr(handler_instance, "do_OPTIONS"):
                handler_instance.do_OPTIONS()
            else:
                send_bytes_response(b"", "text/plain", {
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization"
                })
        else:
            send_bytes_response(b"Method Not Allowed", "text/plain", status_code=405)
    except Exception as exc:
        err_msg = json.dumps({
            "status": "RUNTIME_ERROR",
            "error": str(exc),
            "traceback": traceback.format_exc()
        }, indent=2).encode("utf-8")
        send_bytes_response(err_msg, "application/json", status_code=500)

    status_line = f"{response_status[0]} {response_status[1]}"
    start_response(status_line, response_headers)
    return [output_buffer.getvalue()]


# Base class for BaseHTTPRequestHandler compatibility
if UniversalRequestHandler is not None:
    _Base = UniversalRequestHandler
else:
    _Base = http.server.SimpleHTTPRequestHandler


class VercelUniversalHandler(_Base):
    """Universal handler supporting both BaseHTTPRequestHandler socket invocation
    and WSGI application protocol."""

    def __call__(self, environ, start_response):
        return wsgi_dispatch(environ, start_response)

    def do_GET(self):
        handler_cls = get_handler_class()
        if handler_cls is not None and issubclass(handler_cls, http.server.BaseHTTPRequestHandler):
            return handler_cls.do_GET(self)
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        err_msg = json.dumps({
            "status": "IMPORT_ERROR",
            "error": IMPORT_ERROR,
            "traceback": IMPORT_TRACEBACK
        }, indent=2).encode("utf-8")
        self.wfile.write(err_msg)

    def do_POST(self):
        handler_cls = get_handler_class()
        if handler_cls is not None and issubclass(handler_cls, http.server.BaseHTTPRequestHandler):
            return handler_cls.do_POST(self)
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        err_msg = json.dumps({
            "status": "IMPORT_ERROR",
            "error": IMPORT_ERROR,
            "traceback": IMPORT_TRACEBACK
        }, indent=2).encode("utf-8")
        self.wfile.write(err_msg)


handler = UniversalRequestHandler if UniversalRequestHandler is not None else VercelUniversalHandler
app = handler

if __name__ == "__main__":
    from app.satellite_enhancer import run_server
    run_server()
