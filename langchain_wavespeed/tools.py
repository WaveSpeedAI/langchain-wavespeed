"""LangChain tools backed by the WaveSpeed AI inference platform."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

_CLIENT_NAME = "langchain-wavespeed"
#: Default wait deadline, matching the n8n node. ``None`` would wait forever and
#: strand the calling agent.
_DEFAULT_TIMEOUT = 600.0


class _BaseWaveSpeedTool(BaseTool):
    """Shared configuration for WaveSpeed tools.

    Setup:
        Install ``langchain-wavespeed`` and set the ``WAVESPEED_API_KEY``
        environment variable (get a key at https://wavespeed.ai).

        .. code-block:: bash

            pip install -U langchain-wavespeed
            export WAVESPEED_API_KEY="your-api-key"
    """

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("WAVESPEED_API_KEY", default=None),
        description="WaveSpeed API key. Reads WAVESPEED_API_KEY if not given.",
    )
    timeout: float | None = Field(
        default=_DEFAULT_TIMEOUT,
        description=(
            "Maximum seconds to wait for a prediction. Explicitly pass None to "
            "wait indefinitely; the task keeps running server-side either way."
        ),
    )
    poll_interval: float = Field(
        default=2.0,
        description="Seconds between result polls.",
    )
    client: Any | None = Field(
        default=None,
        exclude=True,
        description="Preconfigured wavespeed.Client (mainly for testing).",
    )

    @model_validator(mode="after")
    def _build_client(self) -> _BaseWaveSpeedTool:
        if self.client is None:
            from wavespeed import Client

            self.client = Client(
                api_key=self.api_key.get_secret_value() if self.api_key else None,
                client_name=_CLIENT_NAME,
                # Never let a host-configured default turn one tool call into a
                # second, separately billed submission.
                max_retries=0,
            )
        return self

    @staticmethod
    def _format_output(output: Any) -> str:
        """Render one platform output as a line an LLM can use."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            url = output.get("url")
            if isinstance(url, str):
                return url
        return json.dumps(output, default=str)

    def _run_model(self, model: str, input: dict[str, Any]) -> str:
        payload = {k: v for k, v in input.items() if v is not None}
        try:
            result = self.client.run(
                model,
                payload,
                timeout=self.timeout,
                poll_interval=self.poll_interval,
            )
        except Exception as e:  # noqa: BLE001 - surfaced to the agent verbatim
            # The SDK's messages already carry "(task_id: ...)" and the platform
            # error text; keep them so a failed paid task stays traceable.
            raise ToolException(f"WaveSpeed model {model!r} failed: {e}") from e
        outputs = result.get("outputs") or []
        if not outputs:
            raise ToolException(f"WaveSpeed model {model!r} returned no outputs.")
        return "\n".join(self._format_output(o) for o in outputs)


class ImageGenerationInput(BaseModel):
    """Input for WaveSpeed image generation."""

    prompt: str = Field(description="Text description of the image to generate.")
    resolution: str | None = Field(
        default=None,
        description=(
            'Output resolution tier: "1k", "1.5k" or "2k". Higher tiers cost more.'
        ),
    )
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            'Aspect ratio of the generated image, e.g. "1:1", "16:9", "9:16", "4:3".'
        ),
    )


class WaveSpeedImageGeneration(_BaseWaveSpeedTool):
    """Tool that generates images with WaveSpeed AI models.

    Instantiate:
        .. code-block:: python

            from langchain_wavespeed import WaveSpeedImageGeneration

            tool = WaveSpeedImageGeneration()  # uses WAVESPEED_API_KEY

    Invoke:
        .. code-block:: python

            url = tool.invoke({"prompt": "A red panda drinking boba tea"})
    """

    name: str = "wavespeed_image_generation"
    description: str = (
        "Generate an image from a text prompt using WaveSpeed AI. "
        "Returns the URL(s) of the generated image(s), one per line."
    )
    args_schema: type[BaseModel] = ImageGenerationInput
    model: str = Field(
        default="bytedance/seedream-v5.0-pro",
        description="WaveSpeed model id to run.",
    )

    def _run(
        self,
        prompt: str,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> str:
        return self._run_model(
            self.model,
            {"prompt": prompt, "resolution": resolution, "aspect_ratio": aspect_ratio},
        )


class VideoGenerationInput(BaseModel):
    """Input for WaveSpeed video generation."""

    prompt: str = Field(description="Text description of the video to generate.")
    duration: int | None = Field(
        default=None,
        description="Video duration in seconds (model-dependent, e.g. 5 or 10).",
    )


class WaveSpeedVideoGeneration(_BaseWaveSpeedTool):
    """Tool that generates videos with WaveSpeed AI models.

    Instantiate:
        .. code-block:: python

            from langchain_wavespeed import WaveSpeedVideoGeneration

            tool = WaveSpeedVideoGeneration()  # uses WAVESPEED_API_KEY

    Invoke:
        .. code-block:: python

            url = tool.invoke({"prompt": "A drone shot over a glacier", "duration": 5})
    """

    name: str = "wavespeed_video_generation"
    description: str = (
        "Generate a video from a text prompt using WaveSpeed AI. "
        "Returns the URL(s) of the generated video(s), one per line."
    )
    args_schema: type[BaseModel] = VideoGenerationInput
    model: str = Field(
        default="wavespeed-ai/minimax-h3/text-to-video",
        description="WaveSpeed model id to run.",
    )

    def _run(
        self,
        prompt: str,
        duration: int | None = None,
        **kwargs: Any,
    ) -> str:
        return self._run_model(self.model, {"prompt": prompt, "duration": duration})


class RunModelInput(BaseModel):
    """Input for running an arbitrary WaveSpeed model."""

    model: str = Field(
        description='WaveSpeed model id, e.g. "wavespeed-ai/z-image/turbo".'
    )
    input: dict[str, Any] = Field(
        description="Input parameters for the model (e.g. {'prompt': '...'})."
    )

    @field_validator("input", mode="before")
    @classmethod
    def _coerce_json_input(cls, value: Any) -> Any:
        """Accept the JSON *string* many models emit instead of a JSON object."""
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"'input' must be a JSON object or a valid JSON string: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError(
                    "'input' must decode to a JSON object, "
                    f"got {type(parsed).__name__}."
                )
            return parsed
        return value


class WaveSpeedRunModel(_BaseWaveSpeedTool):
    """Tool that runs any model on the WaveSpeed AI platform.

    Instantiate:
        .. code-block:: python

            from langchain_wavespeed import WaveSpeedRunModel

            tool = WaveSpeedRunModel()  # uses WAVESPEED_API_KEY

    Invoke:
        .. code-block:: python

            url = tool.invoke(
                {
                    "model": "wavespeed-ai/z-image/turbo",
                    "input": {"prompt": "A lighthouse at dusk"},
                }
            )
    """

    name: str = "wavespeed_run_model"
    description: str = (
        "Run an arbitrary WaveSpeed AI model by id with a JSON input payload. "
        "Browse available models at https://wavespeed.ai/models. "
        "Returns the output URL(s), one per line."
    )
    args_schema: type[BaseModel] = RunModelInput

    def _run(self, model: str, input: dict[str, Any], **kwargs: Any) -> str:
        return self._run_model(model, input)
