from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 180,
        retry_attempts: int = 3,
        retry_backoff: float = 1.5,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff = max(0.0, retry_backoff)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in {408, 409, 429, 500, 502, 503, 504, 524}
        if isinstance(exc, urllib.error.URLError):
            return True
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if "deepseek" in self.base_url.lower() or "deepseek" in self.model.lower():
            body["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                    if not isinstance(decoded, dict):
                        raise RuntimeError("LLM response JSON was not an object")
                    payload = decoded
                    break
            except json.JSONDecodeError as exc:
                raise RuntimeError("LLM request failed: invalid JSON response") from exc
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt + 1 >= self.retry_attempts or not self._retryable(exc):
                    break
                time.sleep(self.retry_backoff * (2**attempt))
        if payload is None:
            error_name = type(last_error).__name__ if last_error else "UnknownError"
            raise RuntimeError(f"LLM request failed after retries: {error_name}") from last_error
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain message content") from exc


def text_client(model_override: str | None = None) -> OpenAICompatibleClient | None:
    key = os.getenv("VTM_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    return OpenAICompatibleClient(
        api_key=key,
        base_url=os.getenv("VTM_LLM_BASE_URL", "https://api.deepseek.com"),
        model=model_override or os.getenv("VTM_LLM_MODEL") or "deepseek-v4-flash",
        retry_attempts=int(os.getenv("VTM_LLM_RETRY_ATTEMPTS", "3")),
        retry_backoff=float(os.getenv("VTM_LLM_RETRY_BACKOFF_SECONDS", "1.5")),
    )


def vision_client() -> OpenAICompatibleClient | None:
    key = os.getenv("VTM_VISION_API_KEY")
    base = os.getenv("VTM_VISION_BASE_URL")
    model = os.getenv("VTM_VISION_MODEL")
    if not (key and base and model):
        return None
    return OpenAICompatibleClient(
        api_key=key,
        base_url=base,
        model=model,
        retry_attempts=int(os.getenv("VTM_VISION_RETRY_ATTEMPTS", "2")),
        retry_backoff=float(os.getenv("VTM_VISION_RETRY_BACKOFF_SECONDS", "1.0")),
    )


def parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate:
        raise ValueError("No JSON object found in model response")
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed


def image_message(path: Path, prompt: str) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ],
    }
