"""LangChain standard unit tests for WaveSpeed tools."""

from typing import Any
from unittest.mock import MagicMock

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_wavespeed import (
    WaveSpeedImageGeneration,
    WaveSpeedRunModel,
    WaveSpeedVideoGeneration,
)


def _params() -> dict[str, Any]:
    client = MagicMock()
    client.run.return_value = {"outputs": ["https://example.com/out.png"]}
    return {"api_key": "test-key", "client": client}


class TestWaveSpeedImageGenerationUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[WaveSpeedImageGeneration]:
        return WaveSpeedImageGeneration

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"prompt": "A red panda drinking boba tea"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"WAVESPEED_API_KEY": "env-key"},
            {"client": MagicMock()},
            {},
        )


class TestWaveSpeedVideoGenerationUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[WaveSpeedVideoGeneration]:
        return WaveSpeedVideoGeneration

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"prompt": "A drone shot over a glacier", "duration": 5}


class TestWaveSpeedRunModelUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[WaveSpeedRunModel]:
        return WaveSpeedRunModel

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "model": "wavespeed-ai/z-image/turbo",
            "input": {"prompt": "A lighthouse at dusk"},
        }
