import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResult:
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    """Abstract LLM client interface."""

    def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Deterministic stub for offline/local-first runs."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
        h = hashlib.sha256((prompt + str(self.seed)).encode("utf-8")).hexdigest()
        payload = {"digest": h[:12], "len": len(prompt)}
        return LLMResult(text=json.dumps(payload, sort_keys=True))


class OpenAIClient(LLMClient):
    """OpenAI Responses API client."""

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package not installed. Install with `pip install openai`.") from exc

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return LLMResult(text=text, raw=getattr(response, "model_dump", lambda: None)())
