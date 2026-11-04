"""Thin, dependency-light client for the Meta Model API (https://api.meta.ai/v1).

The API speaks the OpenAI chat-completions dialect, but Muse Spark is a
*reasoning* model: hidden reasoning tokens are billed against ``max_tokens``.
With a small budget the model can spend everything on thinking and return
``content: null`` with ``finish_reason: "length"``. ``MuseClient.chat`` treats
that case as retryable and grows the budget instead of surfacing an empty
answer to the agent.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class MuseError(RuntimeError):
    """Raised when the Meta Model API returns an unrecoverable error."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ChatResult:
    model: str
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def reasoning_tokens(self) -> int:
        details = self.usage.get("completion_tokens_details") or {}
        return int(details.get("reasoning_tokens") or 0)

    def json(self) -> Any:
        """Parse ``content`` as JSON (for structured-output calls)."""
        if self.content is None:
            raise MuseError("model returned no content to parse")
        text = self.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("\n") + 1 :] if "\n" in text else text
        return json.loads(text)


class MuseClient:
    """Synchronous client. Pass ``transport`` in tests to avoid the network."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        key = api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("META_MODEL_API_KEY")
        if not key:
            raise MuseError("set MODEL_API_KEY (Meta Model API key from dev.meta.ai)")
        self._http = httpx.Client(
            base_url=base_url or os.environ.get("META_MODEL_API_BASE", DEFAULT_BASE_URL),
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            transport=transport,
        )
        self.max_retries = max_retries
        self._sleep = sleep

    # -- low level -----------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            try:
                resp = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:  # network blip
                if attempt >= self.max_retries:
                    raise MuseError(f"network error: {exc}") from exc
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    try:
                        body = resp.json()
                    except ValueError:
                        body = resp.text
                    raise MuseError(
                        f"{method} {path} -> HTTP {resp.status_code}", status=resp.status_code, body=body
                    )
            attempt += 1
            self._sleep(min(8.0, 0.5 * 2**attempt) + random.random() / 4)

    # -- chat ----------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "muse-spark-1.3",
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        max_budget: int = 32768,
        **extra: Any,
    ) -> ChatResult:
        """One chat completion, growing the token budget if reasoning ate it all."""
        budget = max_tokens
        while True:
            payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": budget, **extra}
            if tools:
                payload["tools"] = tools
            if response_format:
                payload["response_format"] = response_format
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            data = self._request("POST", "/chat/completions", json=payload).json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            result = ChatResult(
                model=data.get("model", model),
                content=message.get("content"),
                tool_calls=message.get("tool_calls") or [],
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage") or {},
                raw=data,
            )
            starved = result.finish_reason == "length" and not result.content and not result.tool_calls
            if starved and budget < max_budget:
                budget = min(max_budget, budget * 4)
                continue
            return result

    # -- image ---------------------------------------------------------------------
    def generate_image(self, prompt: str, *, model: str = "muse-image-1.0", n: int = 1) -> list[bytes]:
        """Muse Image. Returns raw image bytes (webp by default)."""
        import base64

        data = self._request(
            "POST", "/images/generations", json={"model": model, "prompt": prompt, "n": n}
        ).json()
        out: list[bytes] = []
        for item in data.get("data") or []:
            if item.get("b64_json"):
                out.append(base64.b64decode(item["b64_json"]))
            elif item.get("url"):
                out.append(self._http.get(item["url"]).content)
        if not out:
            raise MuseError("image response had no data", body=data)
        return out

    # -- voice ---------------------------------------------------------------------
    def transcribe(self, wav: bytes, *, model: str = "muse-voice-transcribe-1.0", mode: str = "PUSH_TO_TALK") -> str:
        """Muse Voice Transcribe on a WAV clip (<= 10 min, <= 32 MB)."""
        request = json.dumps({"model": model, "audioEncoding": "WAV", "mode": mode})
        resp = self._request(
            "POST",
            "/asr/transcribe",
            headers={"Accept": "application/json"},
            files={
                "request": (None, request, "application/json"),
                "audio": ("note.wav", wav, "audio/wav"),
            },
        )
        return (resp.json().get("transcript") or "").strip()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "MuseClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
