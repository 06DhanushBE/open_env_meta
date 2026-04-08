# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the WebHarvest OpenEnv environment.

The environment simulates web data acquisition tasks with tool selection and
structured actions.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class WebharvestAction(Action):
    """Action for the WebHarvest environment."""

    tool: str = Field(
        ..., description="Tool to use: bs4 | browser | api | none"
    )
    command: str = Field(
        ..., description="Command: select_tool | click | scroll | extract_table | extract_items | use_api | wait"
    )
    params: Dict[str, object] = Field(
        default_factory=dict, description="Command parameters"
    )


class WebharvestObservation(Observation):
    """Observation from the WebHarvest environment."""

    task_name: str = Field(default="", description="Active task name")
    task_goal: str = Field(default="", description="Task objective")
    page_html: str = Field(default="", description="Current HTML snapshot")
    rendered_snapshot: str = Field(
        default="", description="Rendered state summary"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP response headers"
    )
    tools_available: List[str] = Field(
        default_factory=list, description="Tools available to the agent"
    )
    extracted_items: List[Dict[str, object]] = Field(
        default_factory=list, description="Items extracted so far"
    )
    step_count: int = Field(default=0, description="Current step count")
    blocked: bool = Field(default=False, description="Whether the agent is blocked")
    last_action_error: Optional[str] = Field(
        default=None, description="Last action error message"
    )
    reward: float = Field(default=0.0, description="Reward from the last action")
    done: bool = Field(default=False, description="Whether the episode is done")
