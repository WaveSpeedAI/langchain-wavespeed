# langchain-wavespeed

LangChain integration for [WaveSpeed AI](https://wavespeed.ai) — run state-of-the-art
image and video generation models from your LangChain agents and chains.

## Installation

```bash
pip install -U langchain-wavespeed
```

Set your API key (get one at [wavespeed.ai](https://wavespeed.ai)):

```bash
export WAVESPEED_API_KEY="your-api-key"
```

## Tools

### WaveSpeedImageGeneration

Generate images from text prompts (defaults to `bytedance/seedream-v5.0-pro`):

```python
from langchain_wavespeed import WaveSpeedImageGeneration

tool = WaveSpeedImageGeneration()
url = tool.invoke(
    {
        "prompt": "A red panda drinking boba tea, studio lighting",
        "resolution": "2k",      # optional: "1k" | "1.5k" | "2k"
        "aspect_ratio": "16:9",  # optional
    }
)
print(url)  # https://.../output.png
```

### WaveSpeedVideoGeneration

Generate videos from text prompts (defaults to `wavespeed-ai/minimax-h3/text-to-video`, the cheap open-weights starting point; pass `model="bytedance/seedance-2.5/text-to-video"` for the highest quality):

```python
from langchain_wavespeed import WaveSpeedVideoGeneration

tool = WaveSpeedVideoGeneration()
url = tool.invoke({"prompt": "A drone shot over a glacier at sunrise", "duration": 5})
```

### WaveSpeedRunModel

Run any model on the WaveSpeed platform by id
(browse the catalog at [wavespeed.ai/models](https://wavespeed.ai/models)):

```python
from langchain_wavespeed import WaveSpeedRunModel

tool = WaveSpeedRunModel()
url = tool.invoke({
    "model": "wavespeed-ai/z-image/turbo",
    "input": {"prompt": "A lighthouse at dusk"},
})
```

## Use with an agent

```python
from langchain.agents import create_agent
from langchain_wavespeed import WaveSpeedImageGeneration, WaveSpeedVideoGeneration

agent = create_agent(
    "openai:gpt-5",
    tools=[WaveSpeedImageGeneration(), WaveSpeedVideoGeneration()],
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Make me a picture of a corgi surfing."}]}
)
```

## Configuration

All tools accept:

| Parameter | Default | Description |
| --- | --- | --- |
| `api_key` | `WAVESPEED_API_KEY` env var | WaveSpeed API key |
| `model` | tool-specific | Model id to run (image/video tools) |
| `timeout` | `600.0` | Max seconds to wait for a prediction (`None` waits forever) |
| `poll_interval` | `2.0` | Seconds between result polls |

When a prediction fails or times out, the tool raises `ToolException` with the
platform's error text and the task id, so a paid task stays traceable (and an
agent can read the failure instead of crashing the run). A timeout only stops
the waiting - the task keeps running server-side.

`await tool.ainvoke(...)` works, but note that the underlying WaveSpeed SDK is
synchronous: LangChain runs the blocking call in a worker thread, so it will not
block your event loop, but it is not natively async I/O.

## License

MIT

---

**[WaveSpeed AI](https://wavespeed.ai/)** — AI image & video generation platform.
Try it in the browser: **[Image generator](https://wavespeed.ai/image-generator)** · **[Video generator](https://wavespeed.ai/video-generator)**
