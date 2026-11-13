import argparse
import hmac
import json
import os
import secrets
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_BODY_SIZE = 1024 * 1024


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


class _GatewayHandler(BaseHTTPRequestHandler):
    token = ""
    upstream = ""
    timeout = 30

    def log_message(self, format, *args):
        pass

    def log_error(self, format, *args):
        pass

    def log_request(self, code="-", size="-"):
        path = urlsplit(self.path).path
        if path not in ("/api/chat", "/api/tags"):
            path = "<other>"
        print(f"{self.command} {path} {code}", flush=True)

    def _respond(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status, message):
        self._respond(status, json.dumps({"error": message}).encode("utf-8"))

    def _authorized(self):
        authorization = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(
            authorization.encode("latin-1", "replace"),
            expected.encode(),
        ):
            self._error(401, "unauthorized")
            return False
        return True

    def _body(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else -1
        except ValueError:
            size = -1
        if size < 0:
            self._error(411, "Content-Length required")
            return None
        if size > MAX_BODY_SIZE:
            self._error(413, "request too large")
            return None
        try:
            body = self.rfile.read(size)
        except (socket.timeout, TimeoutError):
            self.close_connection = True
            self.log_request(408)
            return None
        if len(body) != size:
            self._error(400, "incomplete body")
            return None
        try:
            payload = json.loads(body, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            payload = None
        if not isinstance(payload, dict):
            self._error(400, "JSON object required")
            return None
        return body

    def _forward(self, body=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = Request(
            f"{self.upstream}{self.path}",
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            try:
                with build_opener(_NoRedirect).open(request, timeout=300) as response:
                    status, content = response.status, response.read()
            except HTTPError as exc:
                status, content = exc.code, exc.read()
        except (URLError, TimeoutError, OSError):
            self._error(502, "ollama unavailable")
            return
        self._respond(status, content)

    def do_POST(self):
        if self.path != "/api/chat":
            self._error(404, "not found")
            return
        if not self._authorized():
            return
        body = self._body()
        if body is not None:
            self._forward(body)

    def do_GET(self):
        if self.path != "/api/tags":
            self._error(404, "not found")
            return
        if self._authorized():
            self._forward()

    def __getattr__(self, name):
        if name.startswith("do_"):
            return lambda: self._error(404, "not found")
        raise AttributeError(name)


def make_handler(token: str, upstream: str):
    if not token:
        raise ValueError("Gateway token must not be empty.")
    try:
        parsed = urlparse(upstream)
        valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Upstream must be an http or https URL.")
    return type(
        "GatewayHandler",
        (_GatewayHandler,),
        {"token": token, "upstream": upstream.rstrip("/")},
    )


def main():
    parser = argparse.ArgumentParser(description="Token-protected Ollama gateway")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    configured_token = os.getenv("TERAS_GATEWAY_TOKEN")
    if configured_token is not None and not configured_token.isascii():
        parser.error("TERAS_GATEWAY_TOKEN must contain only ASCII characters")
    token = configured_token or secrets.token_urlsafe(32)
    if not configured_token:
        print(f"Set OLLAMA_API_KEY in Streamlit Secrets to: {token}", flush=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(token, args.upstream)
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
