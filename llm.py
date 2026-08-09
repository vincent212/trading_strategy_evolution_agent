"""
LLM provider abstraction for the mutation operator.

Default: the Anthropic API (needs ANTHROPIC_API_KEY). If LLM_BASE_URL is set, any
OpenAI-compatible endpoint is used instead — local Ollama, Groq, Kimi/Moonshot,
Gemini, OpenRouter — via the `openai` SDK, with no Anthropic key or spend.

Local (Ollama), no key:
    export LLM_BASE_URL=http://localhost:11434/v1
    export LLM_MODEL=qwen2.5-coder:7b
    python run.py --ticker NVDA

Hosted (example: Groq free tier):
    export LLM_BASE_URL=https://api.groq.com/openai/v1
    export LLM_API_KEY=gsk_...
    export LLM_MODEL=llama-3.3-70b-versatile
"""
from __future__ import annotations
import os

_OPUS_NO_TEMP = ("claude-opus-4-7", "claude-opus-4-8")   # sampling params removed here


class _AnthropicClient:
    backend = "anthropic"

    def __init__(self):
        import anthropic
        self._c = anthropic.Anthropic()

    def mutate(self, system, user, model, max_tokens=1500, temperature=None):
        kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if temperature is not None and not model.startswith(_OPUS_NO_TEMP):
            kwargs["temperature"] = temperature
        msg = self._c.messages.create(**kwargs)
        return next((b.text for b in msg.content if b.type == "text"), "")


class _OpenAICompatClient:
    def __init__(self, base_url, api_key, model):
        from openai import OpenAI
        self._c = OpenAI(base_url=base_url, api_key=(api_key or "none"))
        self.model = model
        self.backend = f"openai-compatible ({base_url}, model={model})"

    def mutate(self, system, user, model=None, max_tokens=1500, temperature=None):
        r = self._c.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=(0.8 if temperature is None else temperature),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        if not r.choices:                       # empty/error response -> clean reject
            return ""
        return r.choices[0].message.content or ""


def make_client(default_model: str):
    """Return a client with a uniform .mutate(system, user, model, ...) -> str.
    OpenAI-compatible when LLM_BASE_URL is set, otherwise Anthropic."""
    base = os.environ.get("LLM_BASE_URL")
    if base:
        model = os.environ.get("LLM_MODEL")
        if not model:
            raise RuntimeError(
                "LLM_BASE_URL is set but LLM_MODEL is not. Set LLM_MODEL to a model the "
                "endpoint serves, e.g. 'qwen2.5-coder:7b' (Ollama) or "
                "'llama-3.3-70b-versatile' (Groq).")
        return _OpenAICompatClient(base, os.environ.get("LLM_API_KEY"), model)
    return _AnthropicClient()
