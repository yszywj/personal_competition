"""GLM API client (OpenAI-compatible chat completions) + mock mode.

Configuration comes exclusively from environment variables; no key is ever
stored in code:

    GLM_API_KEY    -- required for real mode (clear error when missing)
    GLM_BASE_URL   -- default https://open.bigmodel.cn/api/paas/v4
    GLM_MODEL      -- default glm-4.5
    GLM_TIMEOUT_S  -- default 300
    GLM_TEMPERATURE-- default 0.2

Mock mode loads a plan JSON file from disk and returns it verbatim as the
model output, flowing through the exact same schema/validator/executor path
as a real response.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class LLMClientError(RuntimeError):
    """Raised when the LLM backend is misconfigured or fails."""


class GLMClient:
    """One-shot chat client. The commander enforces exactly one call per
    episode; this object may be reused across episodes of one run."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 300.0,
        temperature: float = 0.2,
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
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
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
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """One chat completion; returns the raw content string."""

        try:
            import requests
        except ImportError as error:  # pragma: no cover - depends on runtime
            raise LLMClientError(
                "The 'requests' package is required for real GLM calls"
            ) from error
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        self.call_count += 1
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout_s
            )
        except Exception as error:
            raise LLMClientError(f"GLM request failed: {error}") from error
        if response.status_code != 200:
            raise LLMClientError(
                f"GLM API returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise LLMClientError(
                f"GLM API response has an unexpected shape: {response.text[:500]}"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("GLM API returned empty content")
        return content


class MockGLMClient:
    """Offline client: returns a plan file verbatim, through the same pipeline."""

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
        self.last_latency_s = 0.0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        self.call_count += 1
        self.last_latency_s = time.perf_counter() - start
        return self._content


def extract_json_payload(raw: str) -> tuple[object | None, str | None]:
    """Extract the plan JSON object from a raw model response.

    Accepts a bare JSON object or one wrapped in ```json fences. Returns
    (payload, error). Never repairs or rewrites the payload.
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
