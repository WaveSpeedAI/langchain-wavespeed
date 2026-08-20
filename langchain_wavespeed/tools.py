"""LangChain tools backed by the WaveSpeed AI inference platform."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr, model_validator

_CLIENT_NAME = "langchain-wavespeed"


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
        default=None,
        description="Maximum seconds to wait for a prediction (None = no limit).",
    )
    poll_interval: float = Field(
        default=1.0,
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
            )
        return self

    def _run_model(self, model: str, input: dict[str, Any]) -> str:
        payload = {k: v for k, v in input.items() if v is not None}
        result = self.client.run(
            model,
            payload,
            timeout=self.timeout,
            poll_interval=self.poll_interval,
        )
        outputs = result.get("outputs") or []
        if not outputs:
            raise RuntimeError(f"WaveSpeed model {model!r} returned no outputs.")
        return "\n".join(str(o) for o in outputs)


class ImageGenerationInput(BaseModel):
    """Input for WaveSpeed image generation."""

    prompt: str = Field(description="Text description of the image to generate.")
    size: str | None = Field(
        default=None,
        description='Output resolution as "width*height", e.g. "2048*2048".',
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible results (-1 for random).",
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
        size: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        return self._run_model(
            self.model, {"prompt": prompt, "size": size, "seed": seed}
        )


class VideoGenerationInput(BaseModel):
    """Input for WaveSpeed video generation."""

    prompt: str = Field(description="Text description of the video to generate.")
    duration: int | None = Field(
        default=None,
        description="Video duration in seconds (model-dependent, e.g. 5 or 10).",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible results (-1 for random).",
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
        default="bytedance/seedance-2.5/text-to-video",
        description="WaveSpeed model id to run.",
    )

    def _run(
        self,
        prompt: str,
        duration: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        return self._run_model(
            self.model, {"prompt": prompt, "duration": duration, "seed": seed}
        )


class RunModelInput(BaseModel):
    """Input for running an arbitrary WaveSpeed model."""

    model: str = Field(
        description='WaveSpeed model id, e.g. "wavespeed-ai/z-image/turbo".'
    )
    input: dict[str, Any] = Field(
        description="Input parameters for the model (e.g. {'prompt': '...'})."
    )


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
