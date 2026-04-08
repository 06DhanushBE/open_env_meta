---
title: WebHarvest OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# WebHarvest OpenEnv

WebHarvest is a real-world OpenEnv environment for web data acquisition. Agents must select the right tool, navigate simple UI patterns, and extract structured data while avoiding rate limits.

## Tasks (3 levels)

- **static_prices (easy)**: parse a static HTML price table.
- **dynamic_load (medium)**: click **Load More** to reveal data, then extract items.
- **rate_limited (hard)**: use a hidden API endpoint and respect rate limits to collect 100 items.

## Action Space

`WebharvestAction`

- `tool`: `bs4 | browser | api | none`
- `command`: `select_tool | click | scroll | extract_table | extract_items | use_api | wait`
- `params`: command-specific arguments (e.g. selector, seconds)

## Observation Space

`WebharvestObservation`

- `task_name`, `task_goal`
- `page_html`, `rendered_snapshot`, `headers`
- `tools_available`, `extracted_items`
- `step_count`, `blocked`, `last_action_error`

## Reward

Dense, deterministic reward in $[0,1]$:

- Partial credit for each newly extracted item.
- Zero reward for invalid actions.
- Episode ends on success, block, or max steps.

## Quick Start

```python
import asyncio

from webharvest_env import WebharvestAction, WebharvestEnv

async def run():
    with WebharvestEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(result.observation.task_name)

        result = await env.step(
            WebharvestAction(tool="bs4", command="select_tool", params={"tool": "bs4"})
        )
        result = await env.step(
            WebharvestAction(tool="bs4", command="extract_table", params={})
        )
        print(result.observation.extracted_items)

asyncio.run(run())
```

## Build and Run Locally

```bash
docker build -t webharvest_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 webharvest_env-env:latest
```

## Baseline Inference

The required inference script is in `inference.py` at the repo root and follows the hackathon stdout format.

Required env vars:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` (optional, if using `from_docker_image()`)

Run:

```bash
python inference.py
```

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id <username>/webharvest-openenv
```

The deployed space provides:

- Web UI at `/web`
- API docs at `/docs`
- Health check at `/health`
- WebSocket at `/ws`

## Project Structure

```
webharvest_env/
â”œâ”€â”€ client.py
â”œâ”€â”€ models.py
â”œâ”€â”€ openenv.yaml
â”œâ”€â”€ inference.py
â”œâ”€â”€ server/
â”‚   â”œâ”€â”€ app.py
â”‚   â”œâ”€â”€ webharvest_env_environment.py
â”‚   â””â”€â”€ Dockerfile
â””â”€â”€ README.md
```

## Baseline Scores

Using the provided inference script with Llama-3-8B-Instruct:

- **static_prices**: 1.00 (perfect extraction in 2 steps)
- **dynamic_load**: 1.00 (perfect extraction in 3 steps)
- **rate_limited**: 1.00 (perfect extraction in 10 steps, respecting limits)
