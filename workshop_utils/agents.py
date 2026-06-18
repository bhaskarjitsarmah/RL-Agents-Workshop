"""Small reusable agent pieces shared across notebooks.

We keep ONLY the genuinely reusable plumbing here (parsing model output, the
zero-shot baseline). The interesting harness-evolution logic -- Reflexion loops,
skill optimization -- is written inline in each notebook so participants can read
and modify it.
"""

from __future__ import annotations

import re

from .db import SCHEMA_TEXT, run_sql
from .llm import llm, observe

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
    """Return an agent_fn(question) -> sql: a single zero-shot call, no loop.

    This is the pre-agent version from NB0 (Move 2): one shot, text in, text out,
    no ability to notice it failed. Kept for the NB0 build-up. The real baseline
    that later notebooks must beat is `make_agent` below (it adds the loop).
    """

    def agent_fn(question: str) -> str:
        raw = llm(baseline_prompt(question), model=model)
        return extract_sql(raw)

    return agent_fn


REPAIR_SYSTEM = (
    "You are a meticulous SQLite debugging expert. A query failed to execute. "
    "Diagnose the likely cause and return a corrected query. "
    "Output ONLY the corrected SQL inside a ```sql code block -- no explanation."
)


def repair_prompt(question: str, sql: str, error: str, extra: str = "") -> list:
    """Messages asking the model to fix a query that raised a database error."""
    user = f"Database schema:\n{SCHEMA_TEXT}\n"
    if extra:
        user += f"\n{extra}\n"
    user += (
        f"\nQuestion: {question}"
        f"\n\nThis SQL was attempted:\n{sql}"
        f"\n\nIt failed with this database error:\n{error}"
        f"\n\nReturn a corrected SQLite query."
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_agent(model: str | None = None, extra: str = "", max_repairs: int = 2):
    """The text-to-SQL agent built by hand in NB0: brain + tool + execution loop.

    Per question:
      1. ask the LLM (brain) for SQL, given the schema (context C) and any
         injected `extra` block (memory / skills = state S, used from NB2 on);
      2. run it with `run_sql` (the tool T);
      3. if it raises a database error, feed the error back and retry, up to
         `max_repairs` times (the execution loop E). It does NOT retry on a
         clean run that simply returns the "wrong" rows -- catching subtly wrong
         answers needs a real reward (V, NB1) and learning (S, NB2+).

    Returns agent_fn(question) -> sql. This looped agent is the BASELINE every
    later notebook must beat by evolving the harness, never the weights.
    """

    @observe(name="sql_agent")
    def agent_fn(question: str) -> str:
        # @observe groups the initial call + any repair calls into ONE Langfuse
        # trace per question -- that trace IS the agent's trajectory.
        sql = extract_sql(llm(baseline_prompt(question, extra=extra), model=model))
        for _ in range(max_repairs):
            _, err = run_sql(sql)
            if err is None:
                return sql  # executed cleanly -> done
            sql = extract_sql(llm(repair_prompt(question, sql, err, extra), model=model))
        return sql

    return agent_fn
