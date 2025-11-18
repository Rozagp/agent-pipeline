
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import requests


@dataclass
class LLMConfig:
    api_key: Optional[str]
    base_url: str
    model: str
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.avalai.ir/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
        )

    def enabled(self) -> bool:
        return bool(self.api_key)

    #Bc decision_agent calls .ready
    def ready(self) -> bool:
        return self.enabled()


class OpenAICompatLLM:
    """
    Minimal OpenAI-compatible /chat/completions client that always returns JSON.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self._cfg = cfg or LLMConfig.from_env()

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        Call the LLM and return the parsed JSON object from message.content.
        """
        if not self._cfg.enabled():
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL")

        url = self._cfg.base_url.rstrip("/") + "/chat/completions"

        payload = {
            "model": self._cfg.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._cfg.timeout,
            )
            resp.raise_for_status()
            obj = resp.json()
            content = obj["choices"][0]["message"]["content"]
            return json.loads(content)
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")
            code = getattr(e.response, "status_code", "unknown")
            raise RuntimeError(f"OpenAI HTTP {code}: {body}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raw = resp.text if "resp" in locals() else ""
            raise RuntimeError(f"OpenAI response parsing failed: {e}. Raw: {raw}") from e
