"""A backend-agnostic LLM wrapper, a cost meter, and live Langfuse tracing.

Why this exists
---------------
Harness optimization (Reflexion, SkillOpt, evolutionary search) makes MANY LLM
calls. We instrument the single choke point -- this wrapper -- so that:
  * every call is traced to **Langfuse** (the real "collect trajectories" story),
  * the spend is visible after every loop via a local cost meter.

The OpenAI client is the **Langfuse drop-in** (`from langfuse.openai import OpenAI`),
so swapping it here gives every notebook full tracing with zero call-site changes.
It is still OpenAI-compatible: set OPENAI_BASE_URL to point at Azure / a local
vLLM / Ollama / a corporate proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# --- Hard requirement: the real observability stack must be installed. ---
try:
    # Drop-in replacement for `openai`: identical API, every call traced.
    from langfuse.openai import OpenAI
    from langfuse import get_client, observe
except ModuleNotFoundError as e:  # pragma: no cover - setup guard
    raise ModuleNotFoundError(
        "This workshop requires the full tool stack (Langfuse, etc.). Install it with:\n"
        "    pip install -r requirements.txt\n"
        "Then create the accounts and fill .env -- see SETUP.md."
    ) from e

DEFAULT_MODEL = os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini")

# Approximate USD per 1M tokens. Update if you use a different model/endpoint.
PRICING_PER_1M = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}

# Keys every notebook needs (the brain + observability). Notebook-specific keys
# (W&B, Qdrant) are passed to preflight() as extras by the notebook that needs them.
REQUIRED_ENV = ["OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def preflight(*extra_keys: str) -> None:
    """Fail fast (with a clear message) if any required key is missing.

    Call this in the first cell of every notebook. Pass any notebook-specific
    keys as extras, e.g. ``preflight("QDRANT_URL", "QDRANT_API_KEY")``.
    """
    required = REQUIRED_ENV + list(extra_keys)
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Missing required keys in your .env: " + ", ".join(missing) +
            "\nCopy .env.example to .env and fill them in -- see SETUP.md."
        )
    # Confirm Langfuse credentials actually authenticate against the server.
    client = get_client()
    try:
        if hasattr(client, "auth_check") and not client.auth_check():
            raise RuntimeError(
                "Langfuse keys are set but authentication failed. Check the "
                "LANGFUSE_* values and that LANGFUSE_BASE_URL matches your region "
                "(US: https://us.cloud.langfuse.com  EU: https://cloud.langfuse.com)."
            )
    except RuntimeError:
        raise
    except Exception as e:  # network/SDK hiccup -- surface it, don't hide it
        raise RuntimeError(f"Could not reach Langfuse: {e}") from e
    print("preflight OK ->", ", ".join(required))


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
    of {"role", "content"} dicts. The call is traced to Langfuse automatically;
    token usage is recorded on the global METER.
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


def flush() -> None:
    """Flush buffered Langfuse events. Call at the end of a notebook so traces
    are sent before the kernel goes idle."""
    get_client().flush()
