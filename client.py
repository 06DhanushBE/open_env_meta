# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Webharvest Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import WebharvestAction, WebharvestObservation


class WebharvestEnv(
    EnvClient[WebharvestAction, WebharvestObservation, State]
):
    """
    Client for the Webharvest Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with WebharvestEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(WebharvestAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = WebharvestEnv.from_docker_image("webharvest_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(WebharvestAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: WebharvestAction) -> Dict:
        """
        Convert WebharvestAction to JSON payload for step message.

        Args:
            action: WebharvestAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "tool": action.tool,
            "command": action.command,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WebharvestObservation]:
        """
        Parse server response into StepResult[WebharvestObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with WebharvestObservation
        """
        obs_data = payload.get("observation", {})
        observation = WebharvestObservation(
            task_name=obs_data.get("task_name", ""),
            task_goal=obs_data.get("task_goal", ""),
            page_html=obs_data.get("page_html", ""),
            rendered_snapshot=obs_data.get("rendered_snapshot", ""),
            headers=obs_data.get("headers", {}),
            tools_available=obs_data.get("tools_available", []),
            extracted_items=obs_data.get("extracted_items", []),
            step_count=obs_data.get("step_count", 0),
            blocked=obs_data.get("blocked", False),
            last_action_error=obs_data.get("last_action_error"),
            reward=obs_data.get("reward", 0.0),
            done=obs_data.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
