from __future__ import annotations

import openai

from src.config.settings import AppSettings


class LLMClient:
    def __init__(self, settings: AppSettings) -> None:
        if not settings.llm.api_key:
            raise ValueError(f"Missing LLM API key. Set environment variable {settings.llm.api_key_env}.")
        if settings.llm.provider not in {"openai", "openai_compatible"}:
            raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")

        self._client = None
        self._use_legacy_client = not hasattr(openai, "OpenAI")
        if self._use_legacy_client:
            openai.api_key = settings.llm.api_key
            if settings.llm.base_url:
                openai.api_base = settings.llm.base_url
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.llm.api_key,
                base_url=settings.llm.base_url,
            )
        self._settings = settings

    def generate(self, prompt: str) -> str:
        if self._use_legacy_client:
            response = openai.ChatCompletion.create(
                model=self._settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._settings.llm.temperature,
                max_tokens=self._settings.llm.max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
        else:
            response = self._client.chat.completions.create(
                model=self._settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._settings.llm.temperature,
                max_tokens=self._settings.llm.max_tokens,
            )
            content = response.choices[0].message.content
        return (content or "").strip()
