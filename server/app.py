# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Webharvest Env Environment.

This module creates an HTTP server that exposes the WebharvestEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import WebharvestAction, WebharvestObservation
    from server.webharvest_env_environment import WebharvestEnvironment
except ModuleNotFoundError:
    from webharvest_env.models import WebharvestAction, WebharvestObservation
    from webharvest_env.server.webharvest_env_environment import (
        WebharvestEnvironment,
    )


# Create the app with web interface and README integration
app = create_app(
    WebharvestEnvironment,
    WebharvestAction,
    WebharvestObservation,
    env_name="webharvest_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


@app.get("/")
def landing_page():
        """Simple landing page for quick verification in HF Spaces."""
        return """
        <!doctype html>
        <html lang=\"en\">
            <head>
                <meta charset=\"utf-8\" />
                <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
                <title>WebHarvest OpenEnv</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 32px; background: #0f172a; color: #e2e8f0; }
                    .card { max-width: 720px; background: #111827; padding: 24px; border-radius: 12px; border: 1px solid #1f2937; }
                    h1 { margin: 0 0 8px; font-size: 24px; }
                    p { margin: 8px 0; line-height: 1.5; }
                    code { background: #0b1220; padding: 2px 6px; border-radius: 6px; }
                    a { color: #60a5fa; text-decoration: none; }
                </style>
            </head>
            <body>
                <div class=\"card\">
                    <h1>WebHarvest OpenEnv</h1>
                    <p>Environment server is running.</p>
                    <p>Useful endpoints:</p>
                    <p><code>POST /reset</code>, <code>POST /step</code>, <code>GET /state</code>, <code>GET /schema</code></p>
                    <p>API docs: <a href=\"/docs\">/docs</a></p>
                    <p>Health: <a href=\"/health\">/health</a></p>
                </div>
            </body>
        </html>
        """


@app.get("/web")
def web_redirect():
        """Compatibility redirect for HF logs panel."""
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/")


def _run(host: str, port: int) -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m webharvest_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn webharvest_env.server.app:app --workers 4
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
