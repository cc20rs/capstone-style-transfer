from __future__ import annotations

import time
from typing import Any, Dict, Optional

from openai import OpenAI


class APIClient:
    def __init__(self, config: Dict[str, Any], api_key: str) -> None:
        self.base_url = config["api"]["base_url"].rstrip("/")
        self.chat_endpoint = config["api"]["chat_endpoint"]
        self.timeout = config["api"]["timeout_seconds"]
        self.max_retries = config["api"]["max_retries"]
        self.retry_backoff = config["api"]["retry_backoff_seconds"]
        self.api_key = api_key
        base_url_for_client = self.base_url
        if not base_url_for_client.endswith("/v1") and not base_url_for_client.endswith("/v1/"):
            base_url_for_client = f"{base_url_for_client}/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=base_url_for_client)

    def chat_completion(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {}
        if extra_payload:
            payload.update(extra_payload)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **payload,
                )
                message = response.choices[0].message.content or ""
                return message.strip()
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)
        raise RuntimeError(f"API request failed after retries: {last_error}")
