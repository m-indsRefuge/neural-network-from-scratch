"""Local HTTP bridge for the live Neural Observatory."""

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

from neural_network_from_scratch.experiments import (
    Experiment,
    linear_split_experiment,
)
from neural_network_from_scratch.live_observatory_html import (
    render_live_observatory_html,
)
from neural_network_from_scratch.live_training import LiveTrainingSession
from neural_network_from_scratch.live_view import build_live_view_state


class LiveObservatoryServer(HTTPServer):
    """Single-session localhost server for live NN-01 training."""

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        experiment: Experiment,
        resolution: int,
    ) -> None:
        self.experiment = experiment
        self.session = LiveTrainingSession(experiment)
        self.resolution = resolution
        self.viewer_html = render_live_observatory_html(
            experiment,
            resolution=resolution,
        )

        super().__init__(
            server_address,
            LiveObservatoryRequestHandler,
        )


class LiveObservatoryRequestHandler(BaseHTTPRequestHandler):
    """Serve the live viewer and its minimal training API."""

    server: LiveObservatoryServer

    def log_message(
        self,
        _format: str,
        *args: object,
    ) -> None:
        """Suppress default HTTP request logging."""
        return

    def _route_path(self) -> str:
        return urlsplit(self.path).path

    def _write_response(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header(
            "Content-Type",
            content_type,
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)

    def _write_json(
        self,
        status: HTTPStatus,
        payload: dict[str, object],
    ) -> None:
        body = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

        self._write_response(
            status,
            body,
            content_type="application/json; charset=utf-8",
        )

    def _write_html(
        self,
        status: HTTPStatus,
        html: str,
    ) -> None:
        body = html.encode("utf-8")

        self._write_response(
            status,
            body,
            content_type="text/html; charset=utf-8",
        )

    def _write_not_found(self) -> None:
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not found"},
        )

    def _write_method_not_allowed(self) -> None:
        self._write_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"error": "method not allowed"},
        )

    def do_GET(self) -> None:
        """Handle Observatory read requests."""
        path = self._route_path()

        try:
            if path == "/":
                self._write_html(
                    HTTPStatus.OK,
                    self.server.viewer_html,
                )
                return

            if path == "/api/state":
                state = build_live_view_state(
                    self.server.session,
                    self.server.experiment,
                    resolution=self.server.resolution,
                )

                self._write_json(
                    HTTPStatus.OK,
                    state.to_dict(),
                )
                return

            if path == "/api/step":
                self._write_method_not_allowed()
                return

            self._write_not_found()

        except Exception:  # noqa: BLE001
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal server error"},
            )

    def do_POST(self) -> None:
        """Handle exactly one live training mutation."""
        path = self._route_path()

        try:
            if path in {"/", "/api/state"}:
                self._write_method_not_allowed()
                return

            if path != "/api/step":
                self._write_not_found()
                return

            if self.server.session.is_complete:
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "training session is complete"},
                )
                return

            self.server.session.step()

            state = build_live_view_state(
                self.server.session,
                self.server.experiment,
                resolution=self.server.resolution,
            )

            self._write_json(
                HTTPStatus.OK,
                state.to_dict(),
            )

        except Exception:  # noqa: BLE001
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal server error"},
            )


def create_live_server(
    *,
    experiment: Experiment | None = None,
    port: int = 0,
    resolution: int = 101,
) -> LiveObservatoryServer:
    """Create one local-only live Observatory server."""
    resolved_experiment = (
        linear_split_experiment()
        if experiment is None
        else experiment
    )

    return LiveObservatoryServer(
        ("127.0.0.1", port),
        experiment=resolved_experiment,
        resolution=resolution,
    )


def main() -> None:
    """Run the live Observatory until interrupted."""
    with create_live_server() as server:
        host, port = server.server_address[:2]

        print(
            f"NEURAL_OBSERVATORY_URL=http://{host}:{port}/",
            flush=True,
        )

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()