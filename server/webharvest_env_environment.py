# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
WebHarvest OpenEnv Environment Implementation.

Simulates web data acquisition tasks with tool selection, clicks, API use,
and deterministic grading.
"""

from __future__ import annotations

import os
from typing import Dict, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WebharvestAction, WebharvestObservation
except ImportError:
    from models import WebharvestAction, WebharvestObservation


TASKS: Dict[str, Dict[str, object]] = {
    "static_prices": {
        "goal": "Extract the full price table from a static HTML page.",
        "tools": ["bs4", "browser"],
        "max_steps": 6,
        "headers": {"content-type": "text/html"},
        "items": [
            {"item": "Widget A", "price": 12.5},
            {"item": "Widget B", "price": 18.0},
            {"item": "Widget C", "price": 7.25},
        ],
        "html": """
<html>
  <body>
    <h1>Catalog</h1>
    <table id=\"prices\">
      <tr><th>Item</th><th>Price</th></tr>
      <tr><td>Widget A</td><td>$12.50</td></tr>
      <tr><td>Widget B</td><td>$18.00</td></tr>
      <tr><td>Widget C</td><td>$7.25</td></tr>
    </table>
  </body>
</html>
""".strip(),
    },
    "dynamic_load": {
        "goal": "Reveal hidden items by clicking Load More, then extract all items.",
        "tools": ["browser"],
        "max_steps": 10,
        "headers": {"content-type": "text/html", "x-rendered": "true"},
        "items": [
            {"sku": "P-100", "name": "Pro Charger"},
            {"sku": "P-200", "name": "Travel Adapter"},
            {"sku": "P-300", "name": "USB-C Dock"},
            {"sku": "P-400", "name": "Noise Cancelling Headset"},
        ],
    },
    "rate_limited": {
        "goal": "Collect 100 items without triggering rate limits. Discover the hidden API endpoint for efficiency.",
        "tools": ["api", "browser"],
        "max_steps": 14,
        "headers": {
            "content-type": "text/html",
            "x-api-endpoint": "/api/items?page=<n>",
            "x-rate-limit": "1 request / 2 seconds",
        },
        "items": [
            {"id": f"SKU-{i:03d}", "label": f"Item {i:03d}"} for i in range(1, 101)
        ],
    },
}


class WebharvestEnvironment(Environment):
    """WebHarvest environment for tool-driven web data acquisition tasks."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        """Initialize the WebHarvest environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._task_names = list(TASKS.keys())
        self._task_index = -1
        self._selected_tool = "none"
        self._loaded = False
        self._scrolls = 0
        self._blocked = False
        self._cooldown = 0
        self._items_extracted: List[Dict[str, object]] = []
        self._last_action_error: str | None = None
        self._done = False

    def reset(self) -> WebharvestObservation:
        """Reset the environment and rotate tasks unless a task is pinned."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        pinned_task = os.getenv("WEBHARVEST_TASK")
        if pinned_task in TASKS:
            task_name = pinned_task
        else:
            self._task_index = (self._task_index + 1) % len(self._task_names)
            task_name = self._task_names[self._task_index]

        self._current_task_name = task_name
        self._current_task = TASKS[task_name]
        self._selected_tool = "none"
        self._loaded = False
        self._scrolls = 0
        self._blocked = False
        self._cooldown = 0
        self._items_extracted = []
        self._last_action_error = None
        self._done = False

        return self._build_observation()

    def step(self, action: WebharvestAction) -> WebharvestObservation:  # type: ignore[override]
        """Execute one step based on tool-driven actions."""
        if self._done:
            self._last_action_error = "episode_done"
            return self._build_observation()

        self._state.step_count += 1
        self._last_action_error = None

        if self._blocked:
            self._done = True
            return self._build_observation()

        reward = 0.0
        command = action.command
        params = action.params or {}

        # Reward for correct tool selection
        if command == "select_tool":
            tool = params.get("tool") or action.tool
            if tool in self._current_task["tools"]:
                self._selected_tool = str(tool)
                reward += 0.1  # Bonus for correct tool choice
            else:
                self._last_action_error = "tool_unavailable"
                reward -= 0.1

        elif command == "click":
            if self._current_task_name != "dynamic_load":
                self._last_action_error = "no_click_targets"
                reward -= 0.05
            elif self._selected_tool != "browser":
                self._last_action_error = "browser_required"
                reward -= 0.05
            else:
                selector = params.get("selector")
                if selector != "button#load-more":
                    self._last_action_error = "invalid_selector"
                    reward -= 0.05
                else:
                    self._loaded = True
                    reward += 0.2  # Reward for successful interaction

        elif command == "scroll":
            if self._current_task_name != "dynamic_load":
                self._last_action_error = "scroll_not_needed"
                reward -= 0.05
            elif self._selected_tool != "browser":
                self._last_action_error = "browser_required"
                reward -= 0.05
            else:
                self._scrolls += 1
                if self._scrolls >= 2:
                    self._loaded = True
                    reward += 0.15  # Partial reward for scrolling
                else:
                    reward += 0.05  # Small reward for attempt

        elif command == "extract_table":
            reward = self._handle_static_extract(params)

        elif command == "extract_items":
            reward = self._handle_dynamic_extract(params)

        elif command == "use_api":
            reward = self._handle_api_extract(params)

        elif command == "wait":
            seconds = int(params.get("seconds", 1))
            self._cooldown = max(0, self._cooldown - max(1, seconds))
            reward += 0.02  # Small reward for patience

        else:
            self._last_action_error = "unknown_command"
            reward -= 0.1

        max_steps = int(self._current_task["max_steps"])
        if self._blocked:
            self._done = True
        elif len(self._items_extracted) >= len(self._current_task["items"]):
            self._done = True
        elif self._state.step_count >= max_steps:
            self._done = True
            if self._last_action_error is None:
                self._last_action_error = "max_steps_reached"

        # Reward shaping: penalties for inefficiency and errors, bonuses for progress
        if self._last_action_error:
            reward -= 0.02  # Penalty for any error
        step_penalty = 0.005 * max(0, self._state.step_count - 2)  # Penalize extra steps
        reward -= step_penalty
        if self._done and not self._blocked and len(self._items_extracted) > 0:
            efficiency_bonus = max(0, 0.1 - 0.01 * (self._state.step_count - len(self._items_extracted)))
            reward += 0.1 + efficiency_bonus  # Completion bonus + efficiency
        reward = max(0.0, min(1.0, reward))

        obs = self._build_observation()
        obs.reward = round(float(reward), 4)  # type: ignore[attr-defined]
        obs.done = self._done  # type: ignore[attr-defined]
        return obs

    def _handle_static_extract(self, params: Dict[str, object]) -> float:
        if self._current_task_name != "static_prices":
            self._last_action_error = "wrong_task"
            return 0.0
        if self._selected_tool != "bs4":
            self._last_action_error = "bs4_required"
            return 0.0

        items = list(self._current_task["items"])
        limit = int(params.get("limit", len(items)))
        batch = items[: max(0, limit)]
        new_items = self._record_items(batch)
        return self._reward_for(new_items)

    def _handle_dynamic_extract(self, params: Dict[str, object]) -> float:
        if self._current_task_name != "dynamic_load":
            self._last_action_error = "wrong_task"
            return 0.0
        if self._selected_tool != "browser":
            self._last_action_error = "browser_required"
            return 0.0
        if not self._loaded:
            self._last_action_error = "content_not_loaded"
            return 0.0

        items = list(self._current_task["items"])
        limit = int(params.get("limit", len(items)))
        batch = items[: max(0, limit)]
        new_items = self._record_items(batch)
        return self._reward_for(new_items)

    def _handle_api_extract(self, params: Dict[str, object]) -> float:
        if self._current_task_name != "rate_limited":
            self._last_action_error = "wrong_task"
            return 0.0
        if self._selected_tool != "api":
            self._last_action_error = "api_tool_required"
            return 0.0
        if self._cooldown > 0:
            self._blocked = True
            self._last_action_error = "rate_limited"
            return 0.0

        batch_size = int(params.get("batch_size", 20))
        batch_size = max(1, min(50, batch_size))
        total_items = list(self._current_task["items"])
        start = len(self._items_extracted)
        batch = total_items[start : start + batch_size]
        new_items = self._record_items(batch)
        self._cooldown = 2
        return self._reward_for(new_items)

    def _record_items(self, items: List[Dict[str, object]]) -> int:
        new_count = 0
        for item in items:
            if item not in self._items_extracted:
                self._items_extracted.append(item)
                new_count += 1
        return new_count

    def _reward_for(self, new_items: int) -> float:
        total = len(self._current_task["items"])
        if total == 0:
            return 0.0
        return min(1.0, new_items / total)

    def _current_html(self) -> str:
        if self._current_task_name == "static_prices":
            return str(self._current_task["html"])
        if self._current_task_name == "dynamic_load":
            if not self._loaded:
                return (
                    "<html><body><h1>Deals</h1>"
                    "<button id=\"load-more\">Load More</button>"
                    "</body></html>"
                )
            return (
                "<html><body><h1>Deals</h1><ul>"
                + "".join(
                    f"<li>{item['sku']} - {item['name']}</li>"
                    for item in self._current_task["items"]
                )
                + "</ul></body></html>"
            )
        if self._current_task_name == "rate_limited":
            if self._blocked:
                return "<html><body><h1>403 Forbidden</h1></body></html>"
            return (
                "<html><body><h1>Inventory API</h1>"
                "<p>Use the API endpoint from headers.</p>"
                "<!-- Hidden API: Use /api/items?page=1 for direct access -->"
                "<script>console.log('API endpoint: /api/items?page=<page>');</script>"
                "</body></html>"
            )
        return ""

    def _rendered_snapshot(self) -> str:
        if self._current_task_name == "dynamic_load":
            if self._loaded:
                return "rendered: list_visible"
            return "rendered: button_visible"
        if self._current_task_name == "rate_limited":
            return f"rendered: items={len(self._items_extracted)}/100"
        return "rendered: table_visible"

    def _build_observation(self) -> WebharvestObservation:
        headers = dict(self._current_task["headers"])
        return WebharvestObservation(
            task_name=self._current_task_name,
            task_goal=str(self._current_task["goal"]),
            page_html=self._current_html(),
            rendered_snapshot=self._rendered_snapshot(),
            headers=headers,
            tools_available=list(self._current_task["tools"]),
            extracted_items=list(self._items_extracted),
            step_count=self._state.step_count,
            blocked=self._blocked,
            last_action_error=self._last_action_error,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
