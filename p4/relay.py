"""A TCP relay that severs a streaming response mid-flight.

This is how the failure drill produces a *real* transport failure instead of a
mocked exception. It sits in front of a real provider, forwards the request,
copies the response back byte for byte, and hard-shuts both sockets once
``cut_after_bytes`` of response data have passed.

Why a raw socket rather than an HTTP library: an HTTP proxy has to re-frame or
buffer the response to know where it is, and buffering delays or suppresses the
mid-stream failure we are trying to cause. Copying raw bytes and then calling
``shutdown()`` is the closest thing to pulling the cable.

No GPU, no mocking, no monkeypatching -- the client sees a genuine
``APIConnectionError`` from the openai SDK, exactly as it would during a real
provider incident.

The module is import-safe and only listens when ``serve()`` is called, so the
rest of the project can import it without binding a port.
"""

import socket
import ssl
import threading
from typing import Optional, Tuple

try:  # the venv's Python has no system CA trust store
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


class CuttingRelay:
    """Forwards to ``upstream_host:upstream_port`` and cuts the response.

    Args:
        upstream_host: provider host to forward to.
        upstream_port: provider port.
        path: request path to forward to (every request goes to the same path).
        cut_after_bytes: response bytes to pass through before severing.
            ``None`` means never cut, which makes the relay a transparent
            passthrough -- useful for proving the drill's own plumbing is not
            what breaks the connection.
        ca_file: CA bundle for the upstream TLS handshake. Defaults to
            certifi's, because the Homebrew Python used here cannot verify
            against the system store.
    """

    def __init__(
        self,
        upstream_host: str = "openrouter.ai",
        upstream_port: int = 443,
        path: str = "/api/v1/chat/completions",
        cut_after_bytes: Optional[int] = 2200,
        ca_file: Optional[str] = None,
        host: str = "127.0.0.1",
    ):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.path = path
        self.cut_after_bytes = cut_after_bytes
        self.ca_file = ca_file or (certifi.where() if certifi else None)
        self.host = host

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0

        #: Bytes of response data forwarded before the cut.
        self.bytes_forwarded = 0
        #: Requests seen.
        self.requests = 0
        #: Whether a cut actually happened.
        self.cut = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> str:
        """Bind, listen, and serve in a daemon thread. Returns the base URL."""
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, 0))          # ephemeral port
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}/v1"

    def stop(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- internals ----------------------------------------------------------

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                head, body = self._read_request(conn)
                cl = self._content_length(head)
                while len(body) < cl:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    body += chunk
                threading.Thread(
                    target=self._handle, args=(conn, head, body, cl), daemon=True
                ).start()
            except OSError:
                try:
                    conn.close()
                except OSError:
                    pass

    @staticmethod
    def _read_request(conn: socket.socket) -> Tuple[bytes, bytes]:
        buf = b""
        while b"\r\n\r\n" not in buf:
            piece = conn.recv(4096)
            if not piece:
                break
            buf += piece
        if b"\r\n\r\n" not in buf:
            return buf, b""
        head, rest = buf.split(b"\r\n\r\n", 1)
        return head, rest

    @staticmethod
    def _content_length(head: bytes) -> int:
        for line in head.decode(errors="ignore").split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return 0
        return 0

    def _handle(self, conn, head: bytes, body: bytes, cl: int) -> None:
        upstream = None
        try:
            raw = socket.create_connection(
                (self.upstream_host, self.upstream_port), timeout=30
            )
            ctx = ssl.create_default_context(cafile=self.ca_file)
            upstream = ctx.wrap_socket(raw, server_hostname=self.upstream_host)

            auth = ""
            content_type = "application/json"
            accept = "text/event-stream"
            for line in head.decode(errors="ignore").split("\r\n"):
                low = line.lower()
                if low.startswith("authorization:"):
                    auth = line
                elif low.startswith("content-type:"):
                    content_type = line.split(":", 1)[1].strip()
                elif low.startswith("accept:"):
                    accept = line.split(":", 1)[1].strip()

            forwarded = (
                f"POST {self.path} HTTP/1.1\r\n"
                f"Host: {self.upstream_host}\r\n"
                f"{auth}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Accept: {accept}\r\n"
                f"Connection: close\r\n"
                f"Content-Length: {cl}\r\n\r\n"
            ).encode()
            upstream.sendall(forwarded + body)

            self.requests += 1

            # Request direction: plain copy, never cut.
            threading.Thread(
                target=self._pump, args=(conn, upstream, False), daemon=True
            ).start()
            # Response direction: cut partway through.
            self._pump(upstream, conn, True)
        except Exception:
            pass
        finally:
            for s in (conn, upstream):
                try:
                    if s:
                        s.close()
                except OSError:
                    pass

    def _pump(self, src, dst, cut_response: bool) -> None:
        try:
            while True:
                data = src.recv(4096)
                if not data:
                    break
                if cut_response:
                    self.bytes_forwarded += len(data)
                    if (
                        self.cut_after_bytes is not None
                        and self.bytes_forwarded >= self.cut_after_bytes
                    ):
                        self.cut = True
                        print(
                            f"   [relay] *** HARD CUT after ~{self.bytes_forwarded} "
                            f"response bytes ***"
                        )
                        try:
                            dst.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                src.close()
            except OSError:
                pass