"""GLM API client (OpenAI-compatible chat completions) + mock mode.

Configuration comes exclusively from environment variables; no key is ever
stored in code or written to audit artefacts:

    GLM_API_KEY     -- required for real mode (clear error when missing)
    GLM_BASE_URL    -- default https://open.bigmodel.cn/api/paas/v4
    GLM_MODEL       -- default glm-4.5
    GLM_TIMEOUT_S   -- default 300
    GLM_TEMPERATURE -- default 0.2
    GLM_JSON_MODE   -- auto (default) | on | off

Structured output: per the official BigModel "对话补全" API reference the
supported ``response_format`` values are ``text`` and ``json_object``
(``json_schema`` is NOT documented for this endpoint and is therefore not
used).  ``auto`` requests ``json_object`` once and, only when the endpoint
explicitly rejects that parameter, retries the *same planning call* without
it; ``on`` hard-fails instead; ``off`` never sends it.  The fallback retry is
a transport-level retry of a failed request, never a second planning attempt
over produced content: whatever JSON the accepted response contains goes
through the same strict parser and validator with no repair.

Mock mode loads a plan JSON file from disk and returns it verbatim as the
model output through the exact same schema/validator/executor path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class LLMClientError(RuntimeError):
    """Raised when the LLM backend is misconfigured or fails."""


@dataclass(frozen=True)
class LLMResponse:
    """One completed planning call.  Unknown API fields stay ``None``.

    Token counts are reported by the API only; nothing is ever estimated.
    ``raw_metadata`` keeps the response's non-content fields (usage, model,
    id, ...) for audit; credentials are never included.
    """

    content: str
    model: str | None = None
    http_status: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float = 0.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    json_mode_requested: bool = False
    json_mode_fallback: bool = False
    http_request_count: int = 1


JSON_MODE_CHOICES = ("auto", "on", "off")


def _json_mode_from_env() -> str:
    value = os.environ.get("GLM_JSON_MODE", "auto").strip().lower() or "auto"
    if value not in JSON_MODE_CHOICES:
        raise LLMClientError(
            f"GLM_JSON_MODE must be one of {list(JSON_MODE_CHOICES)}, got {value!r}"
        )
    return value


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _metadata_without_choices(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Audit-safe copy of a chat-completion response (no credentials)."""

    try:
        return {
            key: value
            for key, value in payload.items()
            if key != "choices"
        }
    except AttributeError:
        return {}


class GLMClient:
    """One-shot chat client. The commander enforces exactly one planning
    call per episode; this object may be reused across episodes of one run."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 300.0,
        temperature: float = 0.2,
        json_mode: str = "auto",
    ) -> None:
        if not api_key:
            raise LLMClientError(
                "GLM_API_KEY is not set. Export GLM_API_KEY (and optionally "
                "GLM_BASE_URL / GLM_MODEL) or pass --mock-plan for offline "
                "testing."
            )
        if not base_url:
            raise LLMClientError("GLM_BASE_URL must not be empty")
        if not model:
            raise LLMClientError("GLM_MODEL must not be empty")
        if json_mode not in JSON_MODE_CHOICES:
            raise LLMClientError(
                f"json_mode must be one of {list(JSON_MODE_CHOICES)}"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.json_mode = json_mode
        self.call_count = 0

    @classmethod
    def from_env(cls) -> "GLMClient":
        return cls(
            api_key=os.environ.get("GLM_API_KEY", "").strip(),
            base_url=os.environ.get("GLM_BASE_URL", "").strip()
            or "https://open.bigmodel.cn/api/paas/v4",
            model=os.environ.get("GLM_MODEL", "").strip() or "glm-4.5",
            timeout_s=float(os.environ.get("GLM_TIMEOUT_S", "300")),
            temperature=float(os.environ.get("GLM_TEMPERATURE", "0.2")),
            json_mode=_json_mode_from_env(),
        )

    # ------------------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> tuple[int, str]:
        """Issue one HTTP request; return (status_code, body_text)."""

        try:
            import requests
        except ImportError as error:  # pragma: no cover - depends on runtime
            raise LLMClientError(
                "The 'requests' package is required for real GLM calls"
            ) from error
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
        except Exception as error:
            raise LLMClientError(f"GLM request failed: {error}") from error
        return int(response.status_code), response.text

    @staticmethod
    def _rejects_response_format(status: int, body: str) -> bool:
        """Whether the endpoint explicitly rejected the response_format param.

        Deliberately conservative: only transport-level rejections of the
        parameter itself (4xx mentioning response_format / json_object)
        qualify for the auto fallback.
        """

        if 200 <= status < 300:
            return False
        if status < 400 or status >= 500:
            return False
        lowered = body.lower()
        return "response_format" in lowered or "json_object" in lowered

    def chat(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """One planning call; returns the structured response."""

        base_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        wants_json = self.json_mode in ("auto", "on")
        started = time.perf_counter()
        self.call_count += 1

        http_requests = 0
        fallback = False
        status: int | None = None
        body_text = ""
        if wants_json:
            payload = dict(base_payload)
            # Officially documented structured mode for this endpoint.
            payload["response_format"] = {"type": "json_object"}
            status, body_text = self._post(payload)
            http_requests += 1
            if not 200 <= status < 300 and self.json_mode == "auto":
                if self._rejects_response_format(status, body_text):
                    # Transport-level retry of the same planning call without
                    # the unsupported parameter.  No content was produced by
                    # the rejected request, so this is not a plan-repair call.
                    fallback = True
                    status, body_text = self._post(base_payload)
                    http_requests += 1
        else:
            status, body_text = self._post(base_payload)
            http_requests += 1

        latency = time.perf_counter() - started
        if status is None or not 200 <= status < 300:
            if wants_json and not fallback and self._rejects_response_format(
                status or 0, body_text
            ):
                raise LLMClientError(
                    "GLM API rejected response_format=json_object and "
                    "GLM_JSON_MODE=on forbids the plain-text fallback "
                    f"(HTTP {status}): {body_text[:500]}"
            )
            raise LLMClientError(
                f"GLM API returned HTTP {status}: {body_text[:500]}"
            )

        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError as error:
            raise LLMClientError(
                f"GLM API response is not JSON: {body_text[:500]}"
            ) from error
        if not isinstance(payload, Mapping):
            raise LLMClientError(
                f"GLM API response has an unexpected shape: {body_text[:500]}"
            )
        try:
            choices = payload["choices"]
            first = choices[0]
            content = first["message"]["content"]
            finish_reason = first.get("finish_reason")
        except (KeyError, IndexError, TypeError) as error:
            raise LLMClientError(
                f"GLM API response has an unexpected shape: {body_text[:500]}"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("GLM API returned empty content")

        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return LLMResponse(
            content=content,
            model=payload.get("model") if isinstance(
                payload.get("model"), str
            ) else None,
            http_status=int(status) if status is not None else None,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            finish_reason=(
                finish_reason if isinstance(finish_reason, str) else None
            ),
            latency_s=latency,
            raw_metadata=_metadata_without_choices(payload),
            json_mode_requested=wants_json,
            json_mode_fallback=fallback,
            http_request_count=http_requests,
        )


class MockGLMClient:
    """Offline client returning a plan file verbatim via the same LLMResponse."""

    def __init__(self, plan_path: str | Path) -> None:
        path = Path(plan_path).expanduser()
        if not path.is_file():
            raise LLMClientError(f"Mock plan file does not exist: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise LLMClientError(f"Cannot read mock plan file: {error}") from error
        # The file must contain the plan payload itself (optionally wrapped in
        # a code fence, mirroring what a real model might emit).
        self._content = text
        self.plan_path = path
        self.call_count = 0
        self.model = f"mock:{path.name}"
        self.json_mode = "mock"

    def chat(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        start = time.perf_counter()
        self.call_count += 1
        return LLMResponse(
            content=self._content,
            model=self.model,
            http_status=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            finish_reason="mock",
            latency_s=time.perf_counter() - start,
            raw_metadata={"mock": True, "plan_path": str(self.plan_path)},
            json_mode_requested=False,
            json_mode_fallback=False,
            http_request_count=0,
        )


def extract_json_payload(raw: str) -> tuple[object | None, str | None]:
    """Extract the plan JSON object from a raw model response.

    Accepts a bare JSON object or one wrapped in ```json fences -- nothing
    more.  No json5, no quote/bracket repair, no truncation healing, no
    natural-language conversion; anything standard JSON cannot parse is an
    INVALID_PLAN upstream.
    """

    text = (raw or "").strip()
    if not text:
        return None, "model response is empty"
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "model response contains no JSON object"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        return None, f"model response is not valid JSON: {error}"
    return payload, None
