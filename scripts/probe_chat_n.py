from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe whether the configured OpenAI-compatible chat API returns multiple choices for n>1."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default="probe_chat_n")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--message", default="只回复 OK")
    args = parser.parse_args()

    settings = load_settings(args.config, experiment_id=args.experiment_id)
    if not settings.llm.api_key:
        raise ValueError(f"Missing LLM API key. Set environment variable {settings.llm.api_key_env}.")
    if not settings.llm.base_url:
        raise ValueError("Configured llm.base_url is empty. This probe only supports OpenAI-compatible endpoints.")

    url = settings.llm.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm.model_name,
        "messages": [{"role": "user", "content": args.message}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": args.n,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm.api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    print(f"status_code={response.status_code}")
    print(f"request_url={url}")
    print(f"model={settings.llm.model_name}")
    print(f"requested_n={args.n}")

    try:
        data = response.json()
    except ValueError:
        print("response_is_json=false")
        print(response.text)
        return

    choices = data.get("choices", [])
    print(f"returned_choices={len(choices)}")
    for idx, choice in enumerate(choices, start=1):
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        snippet = (content or "").replace("\n", " ")[:120]
        print(f"choice_{idx}_finish_reason={finish_reason}")
        print(f"choice_{idx}_content={snippet}")

    if response.status_code >= 400:
        print("error_body=" + json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
