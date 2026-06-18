"""A thin, backend-agnostic LLM wrapper + a global cost meter.

Why this exists
---------------
Harness optimization (Reflexion, SkillOpt, evolutionary search) makes MANY LLM
calls. Participants bring their own keys, so we make the spend *visible* after
every loop. The wrapper is OpenAI-compatible: set OPENAI_BASE_URL to point at
Azure / a local vLLM / Ollama / a corporate proxy and nothing else changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini")

# Approximate USD per 1M tokens. Update if you use a different model/endpoint.
# These are intentionally simple; the meter is for awareness, not billing.
PRICING_PER_1M = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}


@dataclass
class CostMeter:
    """Tracks calls, tokens, and approximate cost across the whole notebook."""

    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def record(self, model: str, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.in_tokens += in_tok
        self.out_tokens += out_tok
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        m["calls"] += 1
        m["in"] += in_tok
        m["out"] += out_tok

    def cost(self) -> float:
        total = 0.0
        for model, m in self.by_model.items():
            p = PRICING_PER_1M.get(model, {"in": 0.0, "out": 0.0})
            total += m["in"] / 1_000_000 * p["in"]
            total += m["out"] / 1_000_000 * p["out"]
        return total

    def reset(self) -> None:
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.by_model = {}

    def report(self) -> str:
        return (
            f"[cost meter] calls={self.calls}  "
            f"in_tok={self.in_tokens:,}  out_tok={self.out_tokens:,}  "
            f"~${self.cost():.4f}"
        )

    def __str__(self) -> str:  # so `print(METER)` works
        return self.report()


# One global meter shared by every helper. Call METER.reset() between experiments.
METER = CostMeter()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default OpenAI
        _client = OpenAI(base_url=base_url) if base_url else OpenAI()
    return _client


def llm(
    messages,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    meter: CostMeter | None = None,
    **kwargs,
) -> str:
    """Run one chat completion and return the assistant's text.

    `messages` may be a plain string (treated as a single user turn) or a list
    of {"role", "content"} dicts. Token usage is recorded on the global METER.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    model = model or DEFAULT_MODEL
    client = _get_client()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    usage = resp.usage
    (meter or METER).record(
        model,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )
    return resp.choices[0].message.content or ""
