"""A minimal OpenAI-compatible server, used to prove the CPU-only claim.

The project claims it runs on a machine with no GPU and no hosted API key, by
pointing at a local OpenAI-compatible server. This stub is what makes that
testable without installing Ollama or a GPU runtime: it speaks just enough of
`/v1/chat/completions` (streaming and not) and `/v1/models` for the app to run a
real request through the local path.

It is a test fixture, not a mock of the app's own logic. The app has no idea it
is talking to this instead of Ollama, which is the point: the local path is the
same code path a real Ollama install would use.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = (
    "Rate limiting protects an API by capping how much work a single client "
    "can request in a rolling window. Each request consumes tokens from a "
    "bucket, and the bucket refills at a fixed rate. When the bucket is empty "
    "the server returns 429 with a Retry-After header."
).split()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [
                {"id": "stub-model", "object": "model", "owned_by": "local"}]})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            req = {}

        # An embeddings request is easy to tell apart and must not be answered
        # with chat completions.
        if "embeddings" in self.path:
            self._json({"object": "list", "data": [
                {"embedding": [0.05] * 64, "index": 0,
                 "object": "embedding"}], "model": "stub-embed"})
            return

        stream = bool(req.get("stream"))
        model = req.get("model", "stub-model")

        if not stream:
            self._json({
                "id": "stub", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": " ".join(WORDS)}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": len(WORDS),
                          "total_tokens": 12 + len(WORDS)}})
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers()

        def frame(delta, finish=None):
            chunk = {"id": "stub", "object": "chat.completion.chunk",
                     "model": model,
                     "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": finish}]}
            return f"data: {json.dumps(chunk)}\n\n"

        try:
            self.wfile.write(frame({"role": "assistant"}).encode())
            self.wfile.flush()
            # A real provider paces its chunks; matching that keeps the
            # streaming assertions meaningful instead of trivially true.
            for i, word in enumerate(WORDS):
                piece = word if i == 0 else " " + word
                self.wfile.write(frame({"content": piece}).encode())
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(frame({}, finish="stop").encode())
            usage = {"id": "stub", "object": "chat.completion.chunk",
                     "model": model, "choices": [],
                     "usage": {"prompt_tokens": 12,
                               "completion_tokens": len(WORDS),
                               "total_tokens": 12 + len(WORDS)}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream, which is exactly what the relay
            # drill does on purpose.
            pass


def start_stub(port=0):
    """Start the stub on a background thread. Returns (server, base_url)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"
