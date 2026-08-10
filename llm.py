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


class _SubagentClient:
    """Mutation via a Claude Code subagent, bridged over the filesystem.

    .mutate() posts the prompt to a request file and BLOCKS until the Claude Code harness
    writes a response file. The harness (the interactive session) watches for the request,
    spawns an Agent subagent with the prompt, and writes its returned code back. This lets a
    detached run.py use a strong model as the mutation operator with no API key/spend.
    """
    backend = "claude-code-subagent (file handoff)"
    model = "claude-code-subagent"

    def __init__(self):
        self.dir = os.environ.get(
            "SUBAGENT_DIR", os.path.join(os.path.dirname(__file__), "runs"))
        self.req = os.path.join(self.dir, "mutation_request.json")
        self.resp = os.path.join(self.dir, "mutation_response.txt")
        self._seq = 0

    def mutate(self, system, user, model=None, max_tokens=1500, temperature=None):
        import json
        import time
        os.makedirs(self.dir, exist_ok=True)
        self._seq += 1
        for p in (self.req, self.resp):             # clear stale files
            if os.path.exists(p):
                os.remove(p)
        tmp = self.req + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"seq": self._seq, "system": system, "user": user}, f)
        os.replace(tmp, self.req)                    # atomic publish
        waited, deadline = 0.0, float(os.environ.get("SUBAGENT_TIMEOUT", "1800"))
        while waited < deadline:
            if os.path.exists(self.resp):
                with open(self.resp) as f:
                    txt = f.read()
                for p in (self.req, self.resp):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                return txt
            time.sleep(1.0)
            waited += 1.0
        raise RuntimeError("subagent mutation timed out (no response file written)")


def make_client(default_model: str):
    """Return a client with a uniform .mutate(system, user, model, ...) -> str.
    LLM_PROVIDER=subagent -> Claude Code subagent (file handoff);
    LLM_BASE_URL set -> OpenAI-compatible; otherwise Anthropic."""
    if os.environ.get("LLM_PROVIDER") == "subagent":
        return _SubagentClient()
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
