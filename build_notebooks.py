"""Generate NB0, NB1 and NB2 as .ipynb files from plain-text cell definitions.

Run:  python build_notebooks.py
This keeps notebook content diffable and reproducible -- we never hand-edit JSON.
Regenerating CLEARS outputs; re-run the notebooks with an API key to repopulate.
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
# NB0 -- Build Your First Agent (and Meet the Harness)
# ============================================================================

NB0 = [
    md(r"""
# NB0 - Build Your First Agent (and Meet the Harness)

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

Before we can make an agent *improve itself*, we have to build one. By the end of
this notebook you'll have a working **text-to-SQL agent** - and you'll have built
it piece by piece, so the rest of the day has a name for every part we evolve.

The one idea to hold onto: **an agent = a brain + a harness.**
- **Brain** = the LLM weights. We *never* touch them - fine-tuning the brain is
  what needs a GPU, and the whole point of this workshop is to avoid that.
- **Harness** = everything *around* the brain: the prompt/context, the tools, the
  memory, the control loop. This is the part we build now and optimize all day.

We'll assemble the harness in four moves and name each one using the framework
**H = (E, T, C, S, L, V)** = Execution loop, Tool registry, Context manager,
State store, Lifecycle hooks, e**V**aluation interface.
"""),
    code(r"""
# Setup. We run from the notebooks/ folder, so add the repo root to the path.
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from workshop_utils import (
    build_db, run_sql, llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt,
    preflight, flush,
)

preflight()              # hard-require OPENAI + LANGFUSE keys (see SETUP.md)
DB = build_db()          # deterministic rebuild; same data on every machine
print("Database ready at:", DB)
"""),
    md(r"""
> **Observability is on.** From here, *every* `llm()` call is traced to
> **Langfuse** - open your project at the `LANGFUSE_BASE_URL` you configured and
> you'll see each call appear live. That dashboard is how a real team inspects an
> agent's behaviour, cost, and latency; we'll lean on it throughout.

## Move 1 - The brain, alone

The simplest thing we can do is ask the LLM for SQL with **no schema and no
tools**. Watch what happens: it has to *guess* our table and column names. It
might look confident and still be completely wrong - and nothing here can tell.
"""),
    code(r"""
question = "How many customers are in Mumbai?"
raw = llm(f"Write a SQLite query for this request: {question}")
print(raw)
# This is not an agent. It is a single guess, ungrounded in our actual database.
"""),
    md(r"""
## Move 2 - Give the brain context (C)

The model can't know *our* schema unless we put it in the prompt. The
**context manager (C)** is the part of the harness that decides what the brain
sees. `baseline_prompt` just formats `schema + question` into messages.
"""),
    code(r"""
msgs = baseline_prompt(question)
print(msgs[1]["content"])          # the exact context we send the brain
print("\n--- model's SQL ---")
sql = extract_sql(llm(msgs))       # extract_sql pulls the query out of the reply
print(sql)
# Better - now it uses real columns. But it's still just text in, text out:
# if this query is wrong, the agent has no way to find out.
"""),
    md(r"""
## Move 3 - Give it a tool, let it act (T + E)

An agent doesn't just *emit text* - it **acts on the world and observes the
result**. Our world is the database; our **tool (T)** is `run_sql`, which
executes a query and returns either rows or an error string. Calling that tool is
the first turn of the **execution loop (E)**.
"""),
    code(r"""
rows, err = run_sql(sql)
print("error :", err)
print("rows  :", rows)

# Now deliberately break a query to see the feedback the agent will react to:
bad_rows, bad_err = run_sql("SELECT name FROM custmers WHERE city='Mumbai'")
print("\nbroken query -> error:", bad_err)
# The environment talks back. That error is a free learning signal - no labels.
"""),
    md(r"""
## Move 4 - Close the loop (E + L)

Here's the agentic part: if the tool reports an error, feed that error back to
the brain and let it try again. **generate -> execute -> observe -> retry.** That
feedback loop is what turns a single call into an *agent*. (`L`, lifecycle hooks,
is just *when* we stop: here, on success or after N tries.)
"""),
    code(r"""
REPAIR_SYS = ("You are a SQLite expert. A query failed; return a corrected "
              "query inside a ```sql code block.")

def my_first_agent(question, max_repairs=2):
    sql = extract_sql(llm(baseline_prompt(question)))
    for attempt in range(max_repairs + 1):
        rows, err = run_sql(sql)
        if err is None:
            print(f"  [solved on attempt {attempt}]")
            return sql
        print(f"  [attempt {attempt} failed: {err}] -> reflecting and retrying")
        fix = llm([
            {"role": "system", "content": REPAIR_SYS},
            {"role": "user", "content":
                f"Schema:\n{SCHEMA_TEXT}\nQuestion: {question}\n\n"
                f"Failed SQL:\n{sql}\n\nDatabase error: {err}"},
        ])
        sql = extract_sql(fix)
    return sql

print("FINAL:", my_first_agent("How many customers are in Mumbai?"))
"""),
    md(r"""
## What you just built

You built an agent. Map it back to the framework:

| Component | What we built |
|---|---|
| **C** - context manager | the schema, injected into the prompt |
| **T** - tool registry | `run_sql` (execute a query against the DB) |
| **E** - execution loop | generate -> run -> observe -> retry |
| **L** - lifecycle hooks | stop on success, or after N tries |
| **S** - state store | *(empty)* - the agent forgets everything between questions |
| **V** - evaluation interface | *(none yet)* - we can't measure how good it is |

`S` and `V` are blank. That is **not** an accident - filling them in is the whole
workshop. `V` (a reward signal) comes next in NB1; `S` (memory and skills that
persist and compound) is NB2 onward.
"""),
    code(r"""
# We've packaged exactly this agent as `make_agent`, so every later notebook
# reuses the same harness. The `extra=` slot is where memory / skills (S) plug in.
from workshop_utils import make_agent

agent = make_agent()
print(agent("List the names of customers in the Enterprise segment."))
print("\n", METER)
flush()                  # send the buffered traces to Langfuse
# Open Langfuse now: you'll see a `sql_agent` trace per question, with the
# initial call and any repair calls nested inside it - that's the trajectory.
"""),
    md(r"""
## Takeaways

- An **agent = brain + harness**. We froze the brain and built a harness around
  it: context (**C**), a tool (**T**), and an execution loop (**E**).
- The loop already handles *crashes* for free, with no labels - that's real, but
  it's the easy part.
- Two things are still missing: we can't **measure** quality (**V**), and the
  agent has no **memory** (**S**) - every question starts from zero.

**Next - NB1:** build the evaluation interface **V** and measure the agent you
just built. *You can't improve what you can't measure.*
"""),
]


# ============================================================================
# NB1 -- The Evaluation Interface (V) + Measuring Your Agent
# ============================================================================

NB1 = [
    md(r"""
# NB1 - The Evaluation Interface (V) and Measuring Your Agent

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

In **NB0** you built an agent: brain + context (**C**) + tool (**T**) + execution
loop (**E**). The one thing it's missing is a way to know **how good it is**.
That's component **V** - the evaluation interface - and it's the job of this
notebook.

> **Thesis of the day:** *Reflection is the gradient, the skill document is the
> parameter vector, and your eval set is the loss.* No loss -> no learning. So we
> start with the loss.

In this notebook we:
1. Meet the **environment** (the text-to-SQL task) and the eval set.
2. Build the **reward signal** `score_sql` (execution match - objective, automatic).
3. **Measure the agent from NB0** to get a baseline number.
4. Do error analysis - the failures are the raw material every later notebook
   learns from.
"""),
    code(r"""
# Setup. We run from the notebooks/ folder, so add the repo root to the path.
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_agent,
    preflight, flush,
)

preflight()              # hard-require OPENAI + LANGFUSE keys (see SETUP.md)
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
print("gold vs gold  ->", score_sql(t["gold"], t["gold"]))                    # True
print("wrong query   ->", score_sql("SELECT city FROM customers", t["gold"]))  # False
print("syntax error  ->", score_sql("SELECT nope FROM nope", t["gold"]))       # False
"""),
    md(r"""
## 4. Measure the agent from NB0

`make_agent()` is the exact harness you built in NB0 (context **C** + tool **T** +
execution loop **E**). No memory, no examples, no skills - the bare agent. This
is the **baseline**: the number every later notebook must beat *without touching
the weights*, only by evolving the harness.
"""),
    code(r"""
agent = make_agent()      # the NB0 agent: generate -> run -> repair-on-error

# The entire "harness" the brain sees right now is just this prompt:
print(baseline_prompt("How many customers are there in total?")[1]["content"])
"""),
    md(r"""
### Run the agent on the held-out test split

This makes real API calls with **your** key. 16 test tasks; a few cents on
`gpt-4o-mini`. The cost meter prints the spend. (The repair loop may add a call
or two when a query first fails to execute.)
"""),
    code(r"""
METER.reset()
baseline = evaluate(agent, split="test", verbose=True)
print()
print("TEST accuracy:", round(baseline["accuracy"], 3))
print("by level:    ", {k: round(v["acc"], 2) for k, v in baseline["by_level"].items()})
print(METER)
"""),
    md(r"""
## 5. Error analysis - the fuel for self-evolution

Every failure below is a learning signal. Notice the *kind* of failure: the
execution loop already eliminated the crashes, so what remains are queries that
**run fine but return the wrong rows** - subtly wrong joins, a missing
`status='completed'` filter, the wrong `GROUP BY`. Robustness can't catch those;
only *learning* can. That gap is exactly what NB2 onward fills.
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
flush()                  # ship the eval traces to Langfuse
"""),
    md(r"""
## Takeaways

- The **eval interface (V)** is the foundation of self-evolution. Without an
  objective reward, "self-improvement" is just vibes.
- Execution-match is a clean, replayable reward - no GPU, no LLM judge.
- Your **NB0 agent** is now measured. Everything from here on raises that test
  number by changing the *harness*, never the weights.
- The remaining failures aren't crashes - they're *subtly wrong* answers. You
  can't fix those with a retry; the agent has to **learn**.

### Exercise
1. Re-run the baseline with `temperature=0.7` (edit `llm(...)`). Does accuracy
   change? What does that tell you about prompt vs. sampling?
2. Add one new hard question + gold SQL to `workshop_utils/tasks.py` and re-run.

**Next - NB2:** give the agent a memory (**S**) and let it learn from these
failures. *Reflection is the gradient.*
"""),
]


# ============================================================================
# NB2 -- Reflexion: Reflection is the Gradient
# ============================================================================

NB2 = [
    md(r"""
# NB2 - Reflexion: Reflection is the Gradient

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

In NB0 you built the agent; in NB1 you measured it (the reward **V**). The agent
has one glaring weakness: it has **no memory (S)**. Every question starts from
scratch, so it makes the same class of mistake forever.

Now we fix that - with **zero weight updates**. The mechanism is **Reflexion**
(Shinn et al.): the agent attempts tasks, compares its answers to feedback,
writes **lessons in natural language**, stores them, and consults them on future
questions. In harness terms (H = E, T, **C**, **S**, L, V) we are filling in:
- **S (state store)** - lessons that persist across tasks, and
- **C (context)** - we inject the relevant lessons back into the prompt.

This is the seed of the **skill library** we formalize in NB3-NB5: raw experience
-> distilled lessons -> reused on new problems.
"""),
    code(r"""
import sys, os, json
sys.path.insert(0, os.path.abspath(".."))
from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_agent,
    preflight, flush,
)
preflight()               # hard-require OPENAI + LANGFUSE keys (see SETUP.md)
build_db()
agent = make_agent()      # the NB0 agent (no memory yet)

# Recap the NB1 baseline number (recompute if NB1 wasn't run this session).
try:
    baseline_acc = json.load(open("../data/baseline_test.json"))["accuracy"]
except FileNotFoundError:
    baseline_acc = evaluate(agent, split="test")["accuracy"]
print("baseline test accuracy:", round(baseline_acc, 3))
"""),
    md(r"""
## Learn across tasks: distill lessons from the train split

The recipe:
1. Run the agent on each **train** task (where we DO have gold labels).
2. On every failure, distill **one general, reusable lesson** - by comparing the
   wrong SQL to the gold SQL. The lesson must generalize, not memorize the answer.
3. Store the lessons in memory (**S**) and inject them into the prompt (**C**).
4. Measure on **test** as the memory grows - the *learning curve*.

This is Reflexion as **harness evolution**: the weights never change; the agent
gets better because its *state* and *context* got better.
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
    sql = agent(t["question"])
    if not score_sql(sql, t["gold"]):
        lessons.append(distill_lesson(t["question"], sql, t["gold"]))
print("collected", len(lessons), "lessons from", len(train), "train tasks\n")
for l in lessons:
    print("-", l)
print("\n", METER)
"""),
    md(r"""
### Inject the lessons and watch the test number move

`make_agent(extra=...)` reuses the *same* NB0 harness, but now we prepend the
learned lessons to the context (**C**). We grow the memory in steps and re-measure
**test** each time - the "learning curve" of a self-evolving agent.
"""),
    code(r"""
def memory_block(lessons):
    if not lessons:
        return ""
    return ("Lessons learned from past mistakes (apply when relevant):\n" +
            "\n".join("- " + l for l in lessons))

def make_memory_agent(lessons):
    # Same harness as NB0, with memory (S) injected into the context (C).
    return make_agent(extra=memory_block(lessons))

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
plt.axhline(baseline_acc, ls="--", color="gray", label="NB0 agent baseline")
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
flush()                   # ship traces to Langfuse
"""),
    md(r"""
## Takeaways

- **Reflection is the gradient.** The agent improved on held-out test with zero
  weight updates - only the harness (state **S** + context **C**) changed.
- Memory turns a one-shot agent into one that **learns from its own mistakes**.
- Self-evolution can go **backwards** (memory pollution). You need a gate.

### The gap this leaves (-> NB3, NB4)
These lessons are **unstructured free text**, **unvalidated**, and we dump *all*
of them into every prompt. The skill-optimization papers fix exactly this:
- **NB3 (skill lifecycle):** turn raw experience into *structured* skills, retrieve
  only the relevant ones, and learn the 25%-degrade trap + the meta-skill rubric.
- **NB4 (SkillOpt):** treat the skill document as a *trainable parameter* with a
  learning rate, **validation gate**, and momentum - so memory can't pollute.

### Exercise
1. Make `distill_lesson` produce lessons WITHOUT seeing the gold SQL (only the
   execution result). Are the lessons still useful? Why / why not?
2. Instead of injecting *all* lessons, retrieve only the 2-3 most relevant to each
   question (e.g. keyword overlap). Does test accuracy improve? This previews the
   retrieval step of NB3.
"""),
]


# ============================================================================
# NB3 -- The Skill Lifecycle: generate -> extract -> consume
# ============================================================================

NB3 = [
    md(r"""
# NB3 - The Skill Lifecycle: Generate -> Extract -> Consume

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

In NB2 the agent learned - but its memory was **unstructured free text**,
**unvalidated**, and we **dumped all of it** into every prompt. That doesn't
scale and it can backfire. NB3 turns raw experience into a proper **skill
library**: the trainable state of our frozen agent.

We follow the skill **lifecycle** from the Microsoft/Fudan study:

1. **Generate** - run the agent, collect trajectories (which tasks it solved).
2. **Extract** - distill **structured skills**, and crucially *only from
   **successful** experience* (capture what worked, not just what to avoid).
3. **Consume** - **retrieve** the few relevant skills per question and inject them.

Two hard-won disciplines from that study, which we'll *demonstrate*, not just state:
- **~25% of skills can *degrade* performance.** More skills is not better. You
  need a quality **gate**.
- A short **meta-skill rubric** (generality / correctness / actionability) is the
  gate that keeps the junk out.
"""),
    code(r"""
import sys, os, json, re
sys.path.insert(0, os.path.abspath(".."))
from workshop_utils import (
    build_db, load_tasks, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_agent,
    preflight, flush,
)
preflight()               # hard-require OPENAI + LANGFUSE keys (see SETUP.md)
build_db()
agent = make_agent()      # the NB0 agent (no skills yet)

try:
    baseline_acc = json.load(open("../data/baseline_test.json"))["accuracy"]
except FileNotFoundError:
    baseline_acc = evaluate(agent, split="test")["accuracy"]
print("baseline test accuracy:", round(baseline_acc, 3))
"""),
    md(r"""
## 1. What is a "skill"?

Not a sentence - a **structured, retrievable object**: a *name*, a *trigger*
(when to use it), and a general *pattern*. Structure is what lets us retrieve the
right skill later and score its quality.
"""),
    code(r"""
example_skill = {
    "name": "completed_revenue",
    "when_to_use": "the question asks for revenue/sales and only completed orders should count",
    "pattern": "JOIN order_items -> orders -> products; revenue = SUM(order_items.quantity*products.price); filter WHERE orders.status='completed'",
}
for k, v in example_skill.items():
    print(f"{k:>12}: {v}")
"""),
    md(r"""
## 2. Generate - collect trajectories on the train split

Run the agent on train tasks and record what it **solved** vs **missed**. The
skills will come from the *solved* ones.
"""),
    code(r"""
train = [t for t in load_tasks() if t["split"] == "train"]
METER.reset()
successes, failures = [], []
for t in train:
    sql = agent(t["question"])
    rec = {"question": t["question"], "sql": sql, "gold": t["gold"], "level": t["level"]}
    (successes if score_sql(sql, t["gold"]) else failures).append(rec)
print(f"train: {len(successes)} solved, {len(failures)} missed")
print(METER)
"""),
    md(r"""
## 3. Extract - distill skills from SUCCESS (not just failure)

NB2 learned from mistakes ("don't do X"). The study's key finding is that the
*reliable* skills come from **successful** trajectories - they capture a pattern
that actually worked. We also skip the trivial *easy* wins: a `SELECT ... WHERE`
is not worth a skill. We extract from **medium/hard successes** only.
"""),
    code(r"""
SKILL_SYS = (
    "You turn a SOLVED text-to-SQL example into a reusable SKILL for a frozen agent. "
    "Return STRICT JSON with keys: name, when_to_use, pattern. "
    "'when_to_use' is a trigger for the CLASS of questions (no specific values). "
    "'pattern' is the general SQL technique (joins/filters/aggregation), not the literal query. "
    "Generalize; never mention the specific question."
)

def parse_json_obj(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def extract_skill(rec):
    raw = llm([
        {"role": "system", "content": SKILL_SYS},
        {"role": "user", "content":
            "Schema:\n" + SCHEMA_TEXT +
            "\nSolved question: " + rec["question"] +
            "\nCorrect SQL:\n" + rec["sql"] +
            "\n\nReturn the skill as JSON."},
    ])
    d = parse_json_obj(raw)
    if not d or not all(k in d for k in ("name", "when_to_use", "pattern")):
        return None
    return {k: str(d[k]) for k in ("name", "when_to_use", "pattern")}

rich = [r for r in successes if r["level"] in ("medium", "hard")]
METER.reset()
candidate_skills = []
for rec in rich:
    s = extract_skill(rec)
    if s:
        candidate_skills.append(s)
# de-duplicate by name (keep first)
dedup = {}
for s in candidate_skills:
    dedup.setdefault(s["name"], s)
candidate_skills = list(dedup.values())

print(f"extracted {len(candidate_skills)} skills from {len(rich)} medium/hard wins\n")
for s in candidate_skills:
    print(f"- {s['name']}: USE WHEN {s['when_to_use']}")
print("\n", METER)
"""),
    md(r"""
## 4. Consume - retrieve only the relevant skills

Dumping the whole library into every prompt wastes tokens and invites
interference. Instead we **retrieve** the top-k skills whose trigger overlaps the
question (a simple lexical match here; NB5 upgrades to a hierarchical library).
"""),
    code(r"""
def format_skills(skills):
    if not skills:
        return ""
    lines = ["Relevant skills (reusable patterns from past solved tasks):"]
    for s in skills:
        lines.append(f"- {s['name']}: USE WHEN {s['when_to_use']}. PATTERN: {s['pattern']}")
    return "\n".join(lines)

_WORD = re.compile(r"[a-z]+")
def _tokens(text):
    return set(_WORD.findall(text.lower()))

def retrieve(question, skills, k=3):
    q = _tokens(question)
    scored = [(len(q & _tokens(s["name"] + " " + s["when_to_use"])), s) for s in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for ov, s in scored[:k] if ov > 0]

def make_skill_agent(skills, k=3):
    # Same NB0 harness; per question we inject only the retrieved skills (C <- S).
    def agent_fn(question):
        return make_agent(extra=format_skills(retrieve(question, skills, k)))(question)
    return agent_fn

# Sanity check: what does retrieval pick for a revenue question?
demo_q = "Which product generated the most completed revenue?"
print("retrieved for:", demo_q)
for s in retrieve(demo_q, candidate_skills, k=3):
    print("  -", s["name"])
"""),
    md(r"""
## 5. The 25%-degrade trap

Any real extraction pipeline also produces **junk** - over-general or wrong
skills. Here we add two plausible-but-harmful ones (exactly the kind a careless
pipeline would keep) and **dump the whole unvetted pool** into every prompt. Watch
the score fail to improve - or drop below baseline.
"""),
    code(r"""
junk_skills = [
    {"name": "always_limit", "when_to_use": "any question",
     "pattern": "add LIMIT 1 to every query to be safe"},
    {"name": "subquery_everything", "when_to_use": "any aggregation",
     "pattern": "wrap every aggregate in a nested subquery"},
]
unvetted = candidate_skills + junk_skills

def make_dump_agent(skills):
    block = format_skills(skills)            # same big block for every question
    return make_agent(extra=block)

METER.reset()
acc_dump = evaluate(make_dump_agent(unvetted), split="test")["accuracy"]
print("baseline (no skills) :", round(baseline_acc, 3))
print("ALL skills, dumped   :", round(acc_dump, 3), " <- unvetted + no retrieval")
print(METER)
"""),
    md(r"""
## 6. The gate - a 3-dimension meta-skill rubric

Before a skill enters the library, score it on three dimensions (each 0-2):

- **generality** - applies to many future questions, not one special case
- **correctness** - the SQL advice is actually true for this schema
- **actionability** - concrete enough to change the query the agent writes

We admit a skill only if **no dimension is a 0**. The junk skills should fail.
"""),
    code(r"""
RUBRIC_SYS = (
    "You are a strict reviewer of agent SKILLS for a text-to-SQL agent. "
    "Score the skill on three dimensions, each an integer 0-2 (0 bad, 2 great): "
    "generality (applies to many future questions, not one case), "
    "correctness (the SQL advice is actually true for this schema), "
    "actionability (concrete enough to change the query written). "
    "Return STRICT JSON with integer keys generality, correctness, actionability "
    "and a short string key reason."
)

def score_skill(s):
    raw = llm([
        {"role": "system", "content": RUBRIC_SYS},
        {"role": "user", "content":
            "Schema:\n" + SCHEMA_TEXT +
            f"\nSkill:\nname: {s['name']}\nwhen_to_use: {s['when_to_use']}\npattern: {s['pattern']}\n\nScore it."},
    ])
    d = parse_json_obj(raw) or {}
    return {dim: int(d.get(dim, 0) or 0) for dim in ("generality", "correctness", "actionability")}

METER.reset()
curated = []
for s in unvetted:
    sc = score_skill(s)
    ok = min(sc.values()) >= 1
    print(f"{'KEEP' if ok else 'DROP'}  {s['name']:<22} "
          f"g{sc['generality']} c{sc['correctness']} a{sc['actionability']}")
    if ok:
        curated.append(s)
print(f"\nkept {len(curated)}/{len(unvetted)} skills after the gate")
print(METER)
"""),
    md(r"""
## 7. The payoff - structure + gate + retrieval

Now run the **curated** library with **retrieval** and compare the three regimes.
"""),
    code(r"""
METER.reset()
acc_curated = evaluate(make_skill_agent(curated, k=3), split="test")["accuracy"]
print("baseline (no skills)      :", round(baseline_acc, 3))
print("ALL skills dumped         :", round(acc_dump, 3))
print("curated + retrieved (NB3) :", round(acc_curated, 3))
print(METER)
"""),
    code(r"""
import matplotlib.pyplot as plt
labels = ["baseline\n(NB0)", "all skills\ndumped", "curated +\nretrieved"]
vals = [baseline_acc, acc_dump, acc_curated]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, vals, color=["gray", "indianred", "seagreen"])
plt.ylim(0, 1)
plt.ylabel("test accuracy")
plt.title("Skill lifecycle: a gated, retrieved library beats dump-everything")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.show()
flush()                   # ship traces to Langfuse
"""),
    md(r"""
## Takeaways

- A **skill** is structured (name / trigger / pattern), so it can be **retrieved**
  and **scored** - unlike NB2's free-text lessons.
- The lifecycle is **generate -> extract -> consume**, and you **extract from
  successes**, not only failures.
- **More skills is not better.** An unvetted pool degrades; a small **rubric gate**
  (generality / correctness / actionability) keeps the library clean.
- Retrieval means each prompt sees only the few skills it needs.

### The gap this leaves (-> NB4)
We curated with a *one-shot* rubric and a hand-set threshold. We never **optimized**
the library: no learning rate, no held-out **validation gate**, no momentum. NB4
(**SkillOpt**) treats the skill document as a trainable parameter and tunes it the
way you'd train a neural net - so improvement is systematic, not lucky.

### Exercise
1. Lower the gate to "average >= 1" instead of "min >= 1". Do more skills slip
   through, and does test accuracy fall? You just felt the 25%-degrade trap.
2. Replace lexical `retrieve` with embeddings (e.g. cosine over `text-embedding-3-small`).
   Does retrieval quality improve on the harder questions?
"""),
]


if __name__ == "__main__":
    build(os.path.join(NB_DIR, "NB0_build_your_first_agent.ipynb"), NB0)
    build(os.path.join(NB_DIR, "NB1_eval_interface_and_baseline.ipynb"), NB1)
    build(os.path.join(NB_DIR, "NB2_reflexion.ipynb"), NB2)
    build(os.path.join(NB_DIR, "NB3_skill_lifecycle.ipynb"), NB3)
    print("done")
