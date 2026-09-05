"""Unit tests for WaveSpeed tools with a mocked SDK client."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

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
    tool.invoke({"prompt": "a cat", "resolution": "2k", "aspect_ratio": "16:9"})
    args, _ = client.run.call_args
    assert args[1] == {"prompt": "a cat", "resolution": "2k", "aspect_ratio": "16:9"}


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
    assert args[0] == "wavespeed-ai/minimax-h3/text-to-video"
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
    with pytest.raises(ToolException, match="no outputs"):
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


def test_default_timeout_is_bounded() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedImageGeneration(api_key=API_KEY, client=client)
    tool.invoke({"prompt": "a cat"})
    _, kwargs = client.run.call_args
    assert kwargs["timeout"] == 600.0
    assert kwargs["poll_interval"] == 2.0


def test_build_client_passes_api_key_and_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attribution path: what actually reaches the WaveSpeed SDK Client."""
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"outputs": [IMAGE_URL]}

    monkeypatch.setattr("wavespeed.Client", FakeClient)
    tool = WaveSpeedImageGeneration(api_key=API_KEY)

    assert isinstance(tool.client, FakeClient)
    assert captured["api_key"] == API_KEY
    assert captured["client_name"] == "langchain-wavespeed"
    # A retry at this layer would be a second, separately billed submission.
    assert captured["max_retries"] == 0
    assert tool.invoke({"prompt": "a cat"}) == IMAGE_URL


def test_build_client_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    monkeypatch.setattr("wavespeed.Client", FakeClient)
    WaveSpeedRunModel()
    assert captured["api_key"] is None


def test_run_model_accepts_json_string_input() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedRunModel(api_key=API_KEY, client=client)
    result = tool.invoke(
        {"model": "some/model", "input": '{"prompt": "x", "steps": 4}'}
    )
    assert result == IMAGE_URL
    assert client.run.call_args[0][1] == {"prompt": "x", "steps": 4}


def test_run_model_rejects_non_json_string_input() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedRunModel(api_key=API_KEY, client=client)
    with pytest.raises(Exception, match="JSON"):
        tool.invoke({"model": "some/model", "input": "not json at all"})


def test_run_model_rejects_json_scalar_input() -> None:
    client = make_client([IMAGE_URL])
    tool = WaveSpeedRunModel(api_key=API_KEY, client=client)
    with pytest.raises(Exception, match="JSON object"):
        tool.invoke({"model": "some/model", "input": "[1, 2]"})


def test_dict_outputs_are_rendered_as_urls_or_json() -> None:
    client = make_client(
        [{"url": VIDEO_URL, "duration": 5}, {"audio": "no url here"}, IMAGE_URL]
    )
    tool = WaveSpeedRunModel(api_key=API_KEY, client=client)
    result = tool.invoke({"model": "some/model", "input": {"prompt": "x"}})
    assert result.splitlines() == [
        VIDEO_URL,
        '{"audio": "no url here"}',
        IMAGE_URL,
    ]


def test_platform_failure_becomes_tool_exception() -> None:
    client = MagicMock()
    client.run.side_effect = RuntimeError(
        "Prediction failed (task_id: task-9): NSFW content"
    )
    tool = WaveSpeedImageGeneration(api_key=API_KEY, client=client)
    with pytest.raises(ToolException, match=r"task_id: task-9\): NSFW content"):
        tool.invoke({"prompt": "a cat"})


def test_timeout_becomes_tool_exception_with_task_id() -> None:
    client = MagicMock()
    client.run.side_effect = TimeoutError(
        "Prediction timed out after 600 seconds (task_id: task-9)"
    )
    tool = WaveSpeedVideoGeneration(api_key=API_KEY, client=client)
    with pytest.raises(ToolException, match="task_id: task-9"):
        tool.invoke({"prompt": "a drone shot"})
