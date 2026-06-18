"""Shared utilities for the RL-for-LLM-Agents workshop.

Everything here is GPU-free. We never touch model weights. The whole point of
the workshop is to evolve the *harness* around a frozen LLM.

Public API:
    llm(messages, **kw)        -> str         (single chat completion)
    METER                      -> CostMeter    (global call/token/cost tracker)
    build_db(path)             -> sqlite3 path (deterministic toy "shop" DB)
    load_tasks()               -> list[dict]   (NL -> gold SQL eval set)
    score_sql(pred, gold, db)  -> bool         (execution-match correctness)
    evaluate(agent_fn, ...)    -> dict          (run an agent over the eval set)
"""

from .llm import llm, METER, CostMeter
from .db import build_db, load_tasks, score_sql, run_sql, DB_PATH, SCHEMA_TEXT
from .evaluate import evaluate
from .agents import extract_sql, baseline_prompt, make_baseline_agent

__all__ = [
    "llm",
    "METER",
    "CostMeter",
    "build_db",
    "load_tasks",
    "score_sql",
    "run_sql",
    "DB_PATH",
    "SCHEMA_TEXT",
    "evaluate",
    "extract_sql",
    "baseline_prompt",
    "make_baseline_agent",
]
