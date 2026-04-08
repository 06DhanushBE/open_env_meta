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


import json

import gradio as gr
import httpx
from fastapi.responses import HTMLResponse, RedirectResponse


# Create the app with web interface and README integration
app = create_app(
    WebharvestEnvironment,
    WebharvestAction,
    WebharvestObservation,
    env_name="webharvest_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def _reset_env() -> dict:
    try:
        response = httpx.post("http://127.0.0.1:8000/reset", timeout=5)
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def _step_env(action_json: str) -> dict:
    try:
        payload = json.loads(action_json)
    except Exception as exc:
        return {"error": f"invalid_json: {exc}"}
    try:
        response = httpx.post(
            "http://127.0.0.1:8000/step",
            json={"action": payload},
            timeout=5,
        )
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return {"error": response.text}
    except Exception as exc:
        return {"error": str(exc)}


with gr.Blocks() as ui:
    gr.Markdown("# WebHarvest OpenEnv\nSimple controls for reset and step.")
    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Step")
    action_input = gr.Textbox(
        label="Action JSON",
        value='{"tool":"bs4","command":"select_tool","params":{"tool":"bs4"}}',
    )
    step_hint = gr.Markdown(
        "Example step payloads: `{" +
        "\"tool\":\"bs4\",\"command\":\"extract_table\",\"params\":{}}`"
    )
    reset_out = gr.JSON(label="Reset Response")
    step_out = gr.JSON(label="Step Response")
    reset_btn.click(_reset_env, outputs=reset_out)
    step_btn.click(_step_env, inputs=action_input, outputs=step_out)
    gr.Markdown("API docs: /docs | Health: /health")


app = gr.mount_gradio_app(app, ui, path="/ui/")


@app.get("/")
def root_redirect():
        html = """
        <!doctype html>
        <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>WebHarvest OpenEnv</title>
                <style>
                    html, body { height: 100%; margin: 0; }
                    iframe { width: 100%; height: 100%; border: 0; }
                </style>
            </head>
            <body>
                <iframe src="/ui/"></iframe>
            </body>
        </html>
        """
        return HTMLResponse(content=html)


@app.get("/web")
def web_redirect():
    return RedirectResponse(url="/")


@app.get("/ui")
def ui_redirect():
    return RedirectResponse(url="/ui/")


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
