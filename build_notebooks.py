"""Generate NB1 and NB2 as .ipynb files from plain-text cell definitions.

Run:  python build_notebooks.py
This keeps notebook content diffable and reproducible -- we never hand-edit JSON.
"""

import os
import nbformat as nbf

NB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def md(text):
    return ("md", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def build(path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


# ============================================================================
# NB1 -- The Evaluation Interface (V) + the Frozen-Brain Baseline
# ============================================================================

NB1 = [
    md(r"""
# NB1 - The Evaluation Interface (V) and the Frozen-Brain Baseline

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

Recall the agent-harness framework from the intro:

> **H = (E, T, C, S, L, V)** = Execution loop, Tool registry, Context manager,
> State store, Lifecycle hooks, e**V**aluation interface.

We never touch the model weights (the "brain"). We evolve the **harness** around
it. But before anything can *evolve*, we need a **reward signal** - component
**V**. That is the whole job of this notebook.

> **Thesis of the day:** *Reflection is the gradient, the skill document is the
> parameter vector, and your eval set is the loss.* No loss -> no learning. So we
> start with the loss.

In this notebook we:
1. Meet the **environment** (a text-to-SQL task over a toy database).
2. Build the **reward signal** `score_sql` (execution match - objective, automatic).
3. Run a **frozen-brain baseline** agent and measure it.
4. Do error analysis - the failures are the raw material every later notebook learns from.
"""),
    code(r"""
# Setup. We run from the notebooks/ folder, so add the repo root to the path.
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_baseline_agent,
)

DB = build_db()          # deterministic rebuild; same data on every machine
print("Database ready at:", DB)
"""),
    md(r"""
## 1. The environment: a toy "shop" database

The agent's job: translate a natural-language question into a **SQLite SELECT**
that returns the right rows. The schema is small enough to fit in a prompt and
rich enough to need joins, group-bys, subqueries, and date handling.
"""),
    code(r"""
print(SCHEMA_TEXT)
print("Sample customers:", run_sql("SELECT * FROM customers LIMIT 3")[0])
print("Sample orders:   ", run_sql("SELECT * FROM orders LIMIT 3")[0])
print("Sample products: ", run_sql("SELECT * FROM products LIMIT 3")[0])
"""),
    md(r"""
## 2. The eval set (NL -> gold SQL)

40 tasks, labelled `easy | medium | hard`, split into **train** (we may optimize
on these) and **test** (held out - the number we actually trust).

The train/test split is the single most important discipline in the workshop:
**a self-evolving agent must improve on train without ever peeking at test.**
"""),
    code(r"""
from collections import Counter
tasks = load_tasks()
print("total tasks:", len(tasks))
print("split:", dict(Counter(t["split"] for t in tasks)))
print("level:", dict(Counter(t["level"] for t in tasks)))
print()
for t in tasks[:3] + tasks[20:22]:
    print(f"#{t['id']:>2} [{t['split']}/{t['level']}] {t['question']}")
    print(f"     gold: {t['gold']}")
"""),
    md(r"""
## 3. The reward signal V: `score_sql` (execution match)

We do **not** compare SQL strings - there are many correct ways to write the same
query. Instead we *execute* both the predicted and the gold query and compare the
**result sets**. Order matters only when the gold query uses `ORDER BY`.

This is exactly the "decomposed verifiable reward" idea from the ASG-SI paper:
an objective, replayable check, not an LLM's opinion.
"""),
    code(r"""
t = next(x for x in tasks if x["id"] == 1)
print("Q:", t["question"])
print("gold vs gold  ->", score_sql(t["gold"], t["gold"]))                 # True
print("wrong query   ->", score_sql("SELECT city FROM customers", t["gold"]))  # False
print("syntax error  ->", score_sql("SELECT nope FROM nope", t["gold"]))      # False
"""),
    md(r"""
## 4. The frozen-brain baseline

The simplest possible agent: hand the schema + question to the LLM, zero-shot,
and parse out the SQL. **No memory, no examples, no tools, no reflection.** This
is the harness in its most bare-bones form - the number every later notebook
must beat *without touching the weights*.
"""),
    code(r"""
def baseline_agent(question):
    messages = baseline_prompt(question)   # zero-shot: schema + question only
    raw = llm(messages)
    return extract_sql(raw)

# Look at the exact prompt we send (the entire "harness" right now):
print(baseline_prompt("How many customers are there in total?")[1]["content"])
"""),
    md(r"""
### Run the baseline on the held-out test split

This makes real API calls with **your** key. 16 test tasks ~= 16 calls; a few
cents on `gpt-4o-mini`. The cost meter prints the spend.
"""),
    code(r"""
METER.reset()
baseline = evaluate(baseline_agent, split="test", verbose=True)
print()
print("TEST accuracy:", round(baseline["accuracy"], 3))
print("by level:    ", {k: round(v["acc"], 2) for k, v in baseline["by_level"].items()})
print(METER)
"""),
    md(r"""
## 5. Error analysis - the fuel for self-evolution

Every failure below is a learning signal. In NB2 (Reflexion) the agent will
*reflect* on these to repair itself; in NB3-NB4 we distill them into reusable,
validated **skills**. Look for patterns: wrong joins? forgetting
`status='completed'`? missing `GROUP BY`? Those patterns become skills.
"""),
    code(r"""
fails = [r for r in baseline["records"] if not r["correct"]]
print(f"{len(fails)} failures on test\n")
for r in fails:
    print(f"# {r['id']} [{r['level']}] {r['question']}")
    print("  pred:", r["pred"])
    print("  gold:", r["gold"])
    print()
"""),
    code(r"""
# Save the baseline number so later notebooks can show the lift over it.
import json
os.makedirs("../data", exist_ok=True)
with open("../data/baseline_test.json", "w", encoding="utf-8") as f:
    json.dump(
        {"accuracy": baseline["accuracy"],
         "by_level": {k: v["acc"] for k, v in baseline["by_level"].items()}},
        f, indent=2,
    )
print("saved baseline_test.json")
"""),
    md(r"""
## Takeaways

- The **eval interface (V)** is the foundation of self-evolution. Without an
  objective reward, "self-improvement" is just vibes.
- Execution-match is a clean, replayable reward - no GPU, no LLM judge.
- The **frozen-brain baseline** is our reference. Everything from here on raises
  the test number by changing the *harness*, never the weights.

### Exercise
1. Re-run the baseline with `temperature=0.7` (edit `llm(... )`). Does accuracy
   change? What does that tell you about prompt vs. sampling?
2. Add one new hard question + gold SQL to `workshop_utils/tasks.py` and re-run.

**Next - NB2:** give the agent a memory and let it learn from these failures.
*Reflection is the gradient.*
"""),
]


# ============================================================================
# NB2 -- Reflexion: Reflection is the Gradient
# ============================================================================

NB2 = [
    md(r"""
# NB2 - Reflexion: Reflection is the Gradient

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

In NB1 we built the reward signal **V** and a frozen-brain baseline. Now we make
the agent *learn* - with **zero weight updates**. The mechanism is **Reflexion**
(Shinn et al.): the agent attempts a task, observes feedback, writes a reflection
in natural language, and uses it on the next attempt.

In harness terms (H = E, T, C, **S**, L, **V**) we are evolving:
- **C (context)** - we inject reflections / lessons into the prompt, and
- **S (state store)** - we persist those lessons across tasks.

Two experiments:
- **A. Within-task self-repair** - use *execution feedback* (no gold) to fix a
  query on the fly. This is online, test-time self-correction.
- **B. Cross-task reflection memory** - use *train labels* to distill general
  lessons, store them, and carry them into future (test) questions. This is the
  agent getting permanently better. It directly previews the **skill library**
  of NB3-NB5.
"""),
    code(r"""
import sys, os, json
sys.path.insert(0, os.path.abspath(".."))
from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_baseline_agent,
)
build_db()
baseline_agent = make_baseline_agent()

# Recap the NB1 baseline number (recompute if NB1 wasn't run).
try:
    baseline_acc = json.load(open("../data/baseline_test.json"))["accuracy"]
except FileNotFoundError:
    baseline_acc = evaluate(baseline_agent, split="test")["accuracy"]
print("baseline test accuracy:", round(baseline_acc, 3))
"""),
    md(r"""
## Experiment A - Within-task self-repair (execution feedback)

At *test time* we do not have the gold answer. But the **environment** still
gives feedback for free: does the query run? does it error? does it return rows?
The agent reflects on that signal and revises - up to `max_tries` times.

> Reward used here is a weak proxy: "executes AND returns >=1 row". It reliably
> fixes syntax errors, wrong column names, and broken joins. It can't catch a
> query that runs but is subtly wrong - that needs a real label (Experiment B).
"""),
    code(r"""
REFLECT_SYS = (
    "You are a meticulous SQLite debugging expert. You diagnose why a query "
    "failed and produce a corrected query."
)

def reflect_and_fix(question, sql, feedback):
    msgs = [
        {"role": "system", "content": REFLECT_SYS},
        {"role": "user", "content":
            "Database schema:\n" + SCHEMA_TEXT +
            "\nQuestion: " + question +
            "\n\nThis SQL was attempted:\n" + sql +
            "\n\nThe environment reported: " + feedback +
            "\n\nFirst explain the likely cause in one sentence. "
            "Then output a corrected SQLite query in a ```sql code block."},
    ]
    return llm(msgs)

def repair_agent(question, max_tries=3):
    sql = extract_sql(llm(baseline_prompt(question)))
    for _ in range(max_tries):
        rows, err = run_sql(sql)
        ok = (err is None) and (len(rows) > 0)
        if ok:
            return sql
        feedback = err if err else "Query executed but returned 0 rows; a filter or join is probably wrong."
        sql = extract_sql(reflect_and_fix(question, sql, feedback))
    return sql
"""),
    code(r"""
METER.reset()
repair = evaluate(repair_agent, split="test", verbose=True)
print()
print("baseline:", round(baseline_acc, 3), " ->  self-repair:", round(repair["accuracy"], 3))
print(METER)
"""),
    md(r"""
## Experiment B - Cross-task reflection memory

Self-repair fixes *one* task in the moment, then forgets. To make the agent
*permanently* better, we let it learn from the **train split** (where we DO have
labels) and carry the lessons forward.

Pipeline:
1. Run the baseline on each **train** task.
2. On every failure, distill **one general, reusable lesson** (not specific to
   that question) by comparing the wrong SQL to the gold SQL.
3. Store lessons in memory (**S**) and prepend them to the prompt (**C**).
4. Measure on **test** as the memory grows.

This is Reflexion as *harness evolution*. It is also the seed of the skill
lifecycle (generate -> extract -> consume) we formalize in NB3.
"""),
    code(r"""
LESSON_SYS = (
    "You distill a single GENERAL, reusable lesson from a SQL mistake. "
    "The lesson must help on FUTURE, different questions - never mention the "
    "specific question or specific values. Keep it to at most two sentences."
)

def distill_lesson(question, wrong_sql, gold_sql):
    msgs = [
        {"role": "system", "content": LESSON_SYS},
        {"role": "user", "content":
            "Schema:\n" + SCHEMA_TEXT +
            "\nQuestion: " + question +
            "\n\nMy incorrect SQL:\n" + wrong_sql +
            "\n\nThe correct SQL:\n" + gold_sql +
            "\n\nWrite ONE general lesson to avoid this class of mistake."},
    ]
    return llm(msgs).strip()

train = [t for t in load_tasks() if t["split"] == "train"]
lessons = []
METER.reset()
for t in train:
    sql = baseline_agent(t["question"])
    if not score_sql(sql, t["gold"]):
        lessons.append(distill_lesson(t["question"], sql, t["gold"]))
print("collected", len(lessons), "lessons from", len(train), "train tasks\n")
for l in lessons[:6]:
    print("-", l)
print("\n", METER)
"""),
    code(r"""
def memory_block(lessons):
    if not lessons:
        return ""
    return ("Lessons learned from past mistakes (apply when relevant):\n" +
            "\n".join("- " + l for l in lessons))

def make_memory_agent(lessons):
    block = memory_block(lessons)
    def agent_fn(q):
        return extract_sql(llm(baseline_prompt(q, extra=block)))
    return agent_fn

# Measure test accuracy as the memory grows: this is the "learning curve".
checkpoints = sorted(set([0, len(lessons) // 3, 2 * len(lessons) // 3, len(lessons)]))
curve = []
METER.reset()
for k in checkpoints:
    acc = evaluate(make_memory_agent(lessons[:k]), split="test")["accuracy"]
    curve.append((k, acc))
    print("lessons in memory = %2d   test accuracy = %.3f" % (k, acc))
print("\n", METER)
"""),
    code(r"""
import matplotlib.pyplot as plt
xs = [c[0] for c in curve]
ys = [c[1] for c in curve]
plt.figure(figsize=(6, 4))
plt.plot(xs, ys, marker="o", label="reflection memory")
plt.axhline(baseline_acc, ls="--", color="gray", label="frozen-brain baseline")
plt.xlabel("# lessons in memory (state S)")
plt.ylabel("test accuracy")
plt.title("Reflexion: accuracy rises as the harness learns (weights frozen)")
plt.ylim(0, 1)
plt.legend()
plt.show()
"""),
    md(r"""
## Failure mode - memory pollution

Self-evolution is not free of risk. A bad or over-general lesson can *lower*
accuracy. The Microsoft/Fudan study (NB3) found **25% of skill pairings actually
degrade performance**. Here we inject a plausible-but-harmful lesson and watch
the number drop - motivating the **validation gates** of SkillOpt (NB4).
"""),
    code(r"""
bad_lessons = ["To be safe, always add LIMIT 1 to every query.",
               "Always wrap every aggregate in a subquery."]
polluted = evaluate(make_memory_agent(lessons + bad_lessons * 2), split="test")["accuracy"]
print("clean memory:", round(curve[-1][1], 3), " ->  polluted memory:", round(polluted, 3))
"""),
    md(r"""
## Takeaways

- **Reflection is the gradient.** The agent improved on held-out test with zero
  weight updates - only the harness (context **C** + state **S**) changed.
- Two regimes: *online self-repair* (execution feedback, no labels) and
  *cross-task memory* (train labels distilled into reusable lessons).
- Self-evolution can go **backwards** (memory pollution). You need a gate.

### The gap this leaves (-> NB3, NB4)
These lessons are **unstructured free text** and **unvalidated**. The skill-
optimization papers fix exactly this:
- **NB3 (skill lifecycle):** turn raw experience into *structured* skills, and
  learn the 25%-degrade trap + the 3-dimension meta-skill rubric.
- **NB4 (SkillOpt):** treat the skill document as a *trainable parameter* with a
  learning rate, **validation gate**, and momentum - so memory can't pollute.

### Exercise
1. Change `max_tries` in `repair_agent` to 1 and 5. Plot accuracy vs. cost.
2. Make `distill_lesson` produce lessons WITHOUT seeing the gold SQL (only the
   execution error). Are the lessons still useful? Why / why not?
"""),
]


if __name__ == "__main__":
    build(os.path.join(NB_DIR, "NB1_eval_interface_and_baseline.ipynb"), NB1)
    build(os.path.join(NB_DIR, "NB2_reflexion.ipynb"), NB2)
    print("done")
