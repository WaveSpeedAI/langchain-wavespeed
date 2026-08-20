"""Unit tests for WaveSpeed tools with a mocked SDK client."""

from unittest.mock import MagicMock

import pytest

from langchain_wavespeed import (
    WaveSpeedImageGeneration,
    WaveSpeedRunModel,
    WaveSpeedVideoGeneration,
)

API_KEY = "test-key"
IMAGE_URL = "https://example.com/image.png"
VIDEO_URL = "https://example.com/video.mp4"


def make_client(outputs: list[str]) -> MagicMock:
    client = MagicMock()
    client.run.return_value = {"outputs": outputs}
    return client


def test_image_generation_invokes_default_model() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedImageGeneration(api_key=API_KEY, client=client)
    result = tool.invoke({"prompt": "a cat"})
    assert result == IMAGE_URL
    args, kwargs = client.run.call_args
    assert args[0] == "bytedance/seedream-v5.0-pro"
    assert args[1] == {"prompt": "a cat"}


def test_image_generation_forwards_optional_params() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedImageGeneration(api_key=API_KEY, client=client)
    tool.invoke({"prompt": "a cat", "size": "1024*1024", "seed": 42})
    args, _ = client.run.call_args
    assert args[1] == {"prompt": "a cat", "size": "1024*1024", "seed": 42}


def test_image_generation_custom_model() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedImageGeneration(
        api_key=API_KEY, client=client, model="wavespeed-ai/z-image/turbo"
    )
    tool.invoke({"prompt": "a cat"})
    assert client.run.call_args[0][0] == "wavespeed-ai/z-image/turbo"


def test_video_generation_invokes_default_model() -> None:
    client = make_client([VIDEO_URL])
    tool = WaveSpeedVideoGeneration(api_key=API_KEY, client=client)
    result = tool.invoke({"prompt": "a drone shot", "duration": 5})
    assert result == VIDEO_URL
    args, _ = client.run.call_args
    assert args[0] == "bytedance/seedance-2.5/text-to-video"
    assert args[1] == {"prompt": "a drone shot", "duration": 5}


def test_run_model_generic() -> None:
    client = make_client([IMAGE_URL, VIDEO_URL])
    tool = WaveSpeedRunModel(api_key=API_KEY, client=client)
    result = tool.invoke({"model": "some/model", "input": {"prompt": "x", "steps": 4}})
    assert result == f"{IMAGE_URL}\n{VIDEO_URL}"
    args, _ = client.run.call_args
    assert args[0] == "some/model"
    assert args[1] == {"prompt": "x", "steps": 4}


def test_timeout_and_poll_interval_forwarded() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedImageGeneration(
        api_key=API_KEY, client=client, timeout=120.0, poll_interval=2.0
    )
    tool.invoke({"prompt": "a cat"})
    _, kwargs = client.run.call_args
    assert kwargs["timeout"] == 120.0
    assert kwargs["poll_interval"] == 2.0


def test_empty_outputs_raises() -> None:
    client = make_client([])
    tool = WaveSpeedImageGeneration(api_key=API_KEY, client=client)
    with pytest.raises(RuntimeError, match="no outputs"):
        tool.invoke({"prompt": "a cat"})


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAVESPEED_API_KEY", "env-key")
    tool = WaveSpeedImageGeneration(client=make_client([IMAGE_URL]))
    assert tool.api_key is not None
    assert tool.api_key.get_secret_value() == "env-key"


async def test_async_invoke_delegates_to_sync() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedImageGeneration(api_key=API_KEY, client=client)
    result = await tool.ainvoke({"prompt": "a cat"})
    assert result == IMAGE_URL
