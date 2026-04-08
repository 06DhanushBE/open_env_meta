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
import pandas as pd
from fastapi.responses import HTMLResponse, RedirectResponse


# Create the app with web interface and README integration
app = create_app(
    WebharvestEnvironment,
    WebharvestAction,
    WebharvestObservation,
    env_name="webharvest_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Separate env instance for the Gradio UI.
ui_env = WebharvestEnvironment()


def _summarize(result: dict) -> dict:
    try:
        obs = result.get("observation", {})
        extracted = obs.get("extracted_items", [])
        return {
            "task": obs.get("task_name"),
            "steps": obs.get("step_count"),
            "extracted": len(extracted) if isinstance(extracted, list) else 0,
            "reward": result.get("reward"),
            "done": result.get("done"),
        }
    except Exception:
        return {"error": "summary_failed"}


def _history_df(history: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": list(range(1, len(history) + 1)),
            "reward": history,
        }
    )


def _reset_env(task_name: str, history: list[float]) -> tuple[dict, dict, pd.DataFrame, list[float]]:
    try:
        if task_name == "auto":
            if "WEBHARVEST_TASK" in os.environ:
                os.environ.pop("WEBHARVEST_TASK")
        else:
            os.environ["WEBHARVEST_TASK"] = task_name
        obs = ui_env.reset()
        result = {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }
        history = [float(obs.reward)]
        return result, _summarize(result), _history_df(history), history
    except Exception as exc:
        return {"error": str(exc)}, {}, _history_df([]), []


def _step_env(action_json: str, history: list[float]) -> tuple[dict, dict, pd.DataFrame, list[float]]:
    try:
        payload = json.loads(action_json)
    except Exception as exc:
        return {"error": f"invalid_json: {exc}"}, {}, _history_df(history or []), history or []
    try:
        action = WebharvestAction(**payload)
        obs = ui_env.step(action)
        result = {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }
        history = list(history or [])
        history.append(float(obs.reward))
        return result, _summarize(result), _history_df(history), history
    except Exception as exc:
        return {"error": str(exc)}, {}, _history_df(history or []), history or []


with gr.Blocks() as ui:
    gr.Markdown("# WebHarvest OpenEnv\nSimple controls for reset and step.")
    with gr.Row():
        task_selector = gr.Dropdown(
            label="Task",
            choices=["auto", "static_prices", "dynamic_load", "rate_limited"],
            value="auto",
        )
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
    with gr.Row():
        btn_select_bs4 = gr.Button("Select BS4")
        btn_extract_table = gr.Button("Extract Table")
        btn_select_browser = gr.Button("Select Browser")
        btn_click_load = gr.Button("Click Load More")
    with gr.Row():
        btn_extract_items = gr.Button("Extract Items")
        btn_select_api = gr.Button("Select API")
        btn_use_api = gr.Button("Use API")
        btn_wait = gr.Button("Wait 2s")
    reset_out = gr.JSON(label="Reset Response")
    step_out = gr.JSON(label="Step Response")
    metrics_out = gr.JSON(label="Metrics")
    reward_chart = gr.LinePlot(
        label="Reward Over Steps",
        x="step",
        y="reward",
        height=240,
    )
    history_state = gr.State([])
    reset_btn.click(
        _reset_env,
        inputs=[task_selector, history_state],
        outputs=[reset_out, metrics_out, reward_chart, history_state],
    )
    step_btn.click(
        _step_env,
        inputs=[action_input, history_state],
        outputs=[step_out, metrics_out, reward_chart, history_state],
    )
    btn_select_bs4.click(
        lambda: '{"tool":"bs4","command":"select_tool","params":{"tool":"bs4"}}',
        outputs=action_input,
    )
    btn_extract_table.click(
        lambda: '{"tool":"bs4","command":"extract_table","params":{}}',
        outputs=action_input,
    )
    btn_select_browser.click(
        lambda: '{"tool":"browser","command":"select_tool","params":{"tool":"browser"}}',
        outputs=action_input,
    )
    btn_click_load.click(
        lambda: '{"tool":"browser","command":"click","params":{"selector":"button#load-more"}}',
        outputs=action_input,
    )
    btn_extract_items.click(
        lambda: '{"tool":"browser","command":"extract_items","params":{}}',
        outputs=action_input,
    )
    btn_select_api.click(
        lambda: '{"tool":"api","command":"select_tool","params":{"tool":"api"}}',
        outputs=action_input,
    )
    btn_use_api.click(
        lambda: '{"tool":"api","command":"use_api","params":{}}',
        outputs=action_input,
    )
    btn_wait.click(
        lambda: '{"tool":"none","command":"wait","params":{"seconds":2}}',
        outputs=action_input,
    )
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
