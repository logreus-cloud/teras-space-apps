import json
import socket
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pytest

from scripts.ollama_gateway import MAX_BODY_SIZE, main, make_handler


TOKEN = "correct-secret"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


@pytest.fixture
def servers():
    calls = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            calls.append(dict(self.headers))
            if getattr(self.server, "redirect_tags", False):
                body = b'{"redirect":true}'
                self.send_response(302)
                self.send_header("Location", "/api/redirect")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models":[]}')

        def log_message(self, format, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    gateway = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(TOKEN, f"http://127.0.0.1:{upstream.server_port}"),
    )
    gateway_thread = Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    try:
        yield gateway, upstream, upstream_thread, calls
    finally:
        gateway.shutdown()
        gateway.server_close()
        gateway_thread.join()
        if upstream_thread.is_alive():
            upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()


def _request(gateway, path="/api/chat", body=b"{}", token=None, method="POST"):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    request = Request(
        f"http://127.0.0.1:{gateway.server_port}{path}",
        data=body if method == "POST" else None,
        headers=headers,
        method=method,
    )
    try:
        with build_opener(NoRedirect).open(request, timeout=5) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def test_missing_token_does_not_reach_upstream(servers):
    gateway, _, _, calls = servers
    assert _request(gateway)[0] == 401
    assert calls == []


def test_wrong_token_does_not_reach_upstream(servers):
    gateway, _, _, calls = servers
    assert _request(gateway, token="wrong")[0] == 401
    assert calls == []


def test_non_ascii_token_is_rejected(servers):
    gateway, _, _, calls = servers
    assert _request(gateway, token="café")[0] == 401
    assert calls == []


def test_non_ascii_configured_token_exits(monkeypatch):
    monkeypatch.setenv("TERAS_GATEWAY_TOKEN", "café")
    monkeypatch.setattr("sys.argv", ["ollama_gateway.py"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_chat_forwards_body_without_authorization(servers):
    gateway, _, _, calls = servers
    body = json.dumps({"model": "qwen3:8b", "messages": []}).encode()
    assert _request(gateway, body=body, token=TOKEN) == (200, body)
    assert len(calls) == 1
    assert all(name.lower() != "authorization" for name in calls[0])


def test_tags(servers):
    gateway, _, _, calls = servers
    assert _request(gateway, path="/api/tags", token=TOKEN, method="GET") == (
        200,
        b'{"models":[]}',
    )
    assert len(calls) == 1


def test_upstream_redirect_is_returned(servers):
    gateway, upstream, _, calls = servers
    upstream.redirect_tags = True
    assert _request(gateway, path="/api/tags", token=TOKEN, method="GET") == (
        302,
        b'{"redirect":true}',
    )
    assert len(calls) == 1


@pytest.mark.parametrize("path", ["/api/pull", "/api/generate"])
def test_other_paths_are_not_forwarded(servers, path):
    gateway, _, _, calls = servers
    assert _request(gateway, path=path, token=TOKEN)[0] == 404
    assert calls == []


def test_large_body_is_rejected(servers):
    gateway, _, _, calls = servers
    assert _request(gateway, body=b"x" * (MAX_BODY_SIZE + 1), token=TOKEN)[0] == 413
    assert calls == []


def test_invalid_json_is_rejected(servers):
    gateway, _, _, calls = servers
    assert _request(gateway, body=b"not JSON", token=TOKEN)[0] == 400
    assert calls == []


def test_deep_json_is_rejected(servers):
    gateway, _, _, calls = servers
    body = b'{"a":' + b"[" * 200000 + b"0" + b"]" * 200000 + b"}"
    assert _request(gateway, body=body, token=TOKEN)[0] == 400
    assert calls == []


def test_incomplete_body_is_rejected(servers):
    gateway, _, _, calls = servers
    with socket.create_connection(("127.0.0.1", gateway.server_port), timeout=5) as sock:
        sock.sendall(
            b"POST /api/chat HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Authorization: Bearer correct-secret\r\n"
            b"Content-Length: 3\r\n"
            b"\r\n"
            b"{}"
        )
        sock.shutdown(socket.SHUT_WR)
        response = HTTPResponse(sock)
        response.begin()
        assert response.status == 400
        response.read()
    assert calls == []


def test_unavailable_upstream(servers):
    gateway, upstream, upstream_thread, _ = servers
    upstream.shutdown()
    upstream.server_close()
    upstream_thread.join()
    status, body = _request(gateway, token=TOKEN)
    assert status == 502
    assert json.loads(body) == {"error": "ollama unavailable"}
