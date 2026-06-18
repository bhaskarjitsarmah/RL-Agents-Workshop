"""Small reusable agent pieces shared across notebooks.

We keep ONLY the genuinely reusable plumbing here (parsing model output, the
zero-shot baseline). The interesting harness-evolution logic -- Reflexion loops,
skill optimization -- is written inline in each notebook so participants can read
and modify it.
"""

from __future__ import annotations

import re

from .db import SCHEMA_TEXT
from .llm import llm

# Matches a ```sql ... ``` or ``` ... ``` fenced block.
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model response.

    Handles fenced code blocks, a leading 'SQL:' label, and trailing prose.
    Returns the first statement (up to the first ';' if present).
    """
    text = (text or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # Drop a leading "SQL:" style label if present.
    text = re.sub(r"^\s*sql\s*:\s*", "", text, flags=re.IGNORECASE)
    # Keep only up to the first complete statement.
    if ";" in text:
        text = text.split(";")[0] + ";"
    return text.strip()


BASELINE_SYSTEM = (
    "You are a precise text-to-SQL assistant for a SQLite database. "
    "Return ONE valid SQLite SELECT query that answers the user's question. "
    "Output ONLY the SQL inside a ```sql code block -- no explanation."
)


def baseline_prompt(question: str, extra: str = "") -> list:
    """Build the zero-shot baseline messages for a question.

    `extra` is an optional block (memory / skills) injected by later notebooks;
    the baseline passes "".
    """
    user = f"Database schema:\n{SCHEMA_TEXT}\n"
    if extra:
        user += f"\n{extra}\n"
    user += f"\nQuestion: {question}\nSQL:"
    return [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_baseline_agent(model: str | None = None):
    """Return an agent_fn(question) -> sql implementing the frozen-brain baseline."""

    def agent_fn(question: str) -> str:
        raw = llm(baseline_prompt(question), model=model)
        return extract_sql(raw)

    return agent_fn
