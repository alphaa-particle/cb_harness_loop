"""Helpers shared by the test files."""

import json


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


async def asgi_call(app, path, payload=None, raw=False):
    """Send one request straight to the app (GET without a payload); return (status, JSON body or text)."""
    body, sent = json.dumps(payload or {}).encode(), []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(event):
        sent.append(event)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "method": "POST" if payload is not None else "GET", "scheme": "http", "path": path,
               "raw_path": path.encode(), "query_string": b"", "root_path": "",
               "headers": [(b"content-type", b"application/json")],
               "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000)}, receive, send)
    status = next(e["status"] for e in sent if e["type"] == "http.response.start")
    body = b"".join(e.get("body", b"") for e in sent if e["type"] == "http.response.body")
    return status, (body.decode("utf-8") if raw else json.loads(body))
