"""Generate NB0-NB6 as .ipynb files from plain-text cell definitions.

Run:  python build_notebooks.py
This keeps notebook content diffable and reproducible -- we never hand-edit JSON.
Regenerating CLEARS outputs of ALL notebooks; re-run them with an API key to
repopulate. To rebuild just one, import this module and call build() for it.
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
"""),
    md(r"""
## Meet the database first

Before we ask the model for any SQL, let's *look* at the database it will be
querying - so the table and column names mean something to you, and you can spot
when the model gets them wrong. It's a small toy **"shop"**: customers place
orders, each order has line items, and each line item points at a product.
"""),
    code(r"""
# A quick tour of the four tables: the schema, then row counts + a few sample rows.
print(SCHEMA_TEXT)

def peek(table, n=4):
    rows, _ = run_sql(f"SELECT * FROM {table} LIMIT {n}")
    total, _ = run_sql(f"SELECT COUNT(*) FROM {table}")
    print(f"\n{table}  -  {total[0][0]} rows total (showing {len(rows)}):")
    for r in rows:
        print("   ", r)

for t in ("customers", "products", "orders", "order_items"):
    peek(t)
"""),
    md(r"""
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
preflight("WANDB_API_KEY")  # OPENAI + LANGFUSE + W&B keys (see SETUP.md)
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

We treat this exactly like a model training run and log it to **Weights &
Biases**: each added lesson is a "step", and the held-out test accuracy is the
metric. This is the same pattern you'd use to track a real fine-tune - here the
"training" is happening in text space.
"""),
    code(r"""
import wandb

def memory_block(lessons):
    if not lessons:
        return ""
    return ("Lessons learned from past mistakes (apply when relevant):\n" +
            "\n".join("- " + l for l in lessons))

def make_memory_agent(lessons):
    # Same harness as NB0, with memory (S) injected into the context (C).
    return make_agent(extra=memory_block(lessons))

# Open a W&B run; lessons_in_memory is our x-axis (the "training step").
run = wandb.init(project="rl-agents-workshop", name="nb2-reflexion",
                 config={"model": "gpt-4o-mini", "n_train": len(train),
                         "n_lessons": len(lessons)})
wandb.define_metric("lessons_in_memory")
wandb.define_metric("test_accuracy", step_metric="lessons_in_memory")

checkpoints = sorted(set([0, len(lessons) // 3, 2 * len(lessons) // 3, len(lessons)]))
curve = []
METER.reset()
for k in checkpoints:
    acc = evaluate(make_memory_agent(lessons[:k]), split="test")["accuracy"]
    curve.append((k, acc))
    wandb.log({"lessons_in_memory": k, "test_accuracy": acc})
    print("lessons in memory = %2d   test accuracy = %.3f" % (k, acc))

wandb.summary["baseline_acc"] = baseline_acc
wandb.summary["final_acc"] = curve[-1][1]
wandb.summary["est_cost_usd"] = METER.cost()
print("\n", METER)
print("W&B run:", run.url)
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

wandb.summary["polluted_acc"] = polluted      # record the failure mode on the run
wandb.finish()                                # close the W&B run
flush()                                       # ship traces to Langfuse
"""),
    md(r"""
## Takeaways

- **Reflection is the gradient.** The agent improved on held-out test with zero
  weight updates - only the harness (state **S** + context **C**) changed.
- Memory turns a one-shot agent into one that **learns from its own mistakes**.
- Self-evolution can go **backwards** (memory pollution). You need a gate.
- We tracked the whole thing in **Weights & Biases** like any training run - the
  learning curve, the baseline, the cost, and the pollution result all live on one
  run you can share and compare. NB4 reuses this to track SkillOpt.

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
preflight("QDRANT_URL", "QDRANT_API_KEY")  # + OPENAI + LANGFUSE (see SETUP.md)
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
## 4. Consume - retrieve relevant skills from a vector DB (Qdrant)

Dumping the whole library into every prompt wastes tokens and invites
interference. A real agent stores skills in a **vector database** and retrieves
only the few relevant to each question. We use **Qdrant Cloud** + OpenAI
embeddings - the exact retrieval (RAG) pattern you'd run in production.
"""),
    code(r"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from workshop_utils import embed          # OpenAI embeddings, traced by Langfuse

COLLECTION = "workshop_skills"
EMBED_DIM = 1536                           # text-embedding-3-small

qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

def format_skills(skills):
    if not skills:
        return ""
    lines = ["Relevant skills (reusable patterns from past solved tasks):"]
    for s in skills:
        lines.append(f"- {s['name']}: USE WHEN {s['when_to_use']}. PATTERN: {s['pattern']}")
    return "\n".join(lines)

def skill_text(s):                         # what we embed: the trigger + pattern
    return f"{s['name']}. USE WHEN {s['when_to_use']}. {s['pattern']}"

def index_skills(skills, collection=COLLECTION):
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
    vectors = embed([skill_text(s) for s in skills])
    qdrant.upsert(collection, points=[
        PointStruct(id=i, vector=v, payload=s)
        for i, (v, s) in enumerate(zip(vectors, skills))])
    return qdrant.count(collection).count

def retrieve(question, k=3, collection=COLLECTION):
    hits = qdrant.query_points(collection, query=embed(question), limit=k).points
    return [h.payload for h in hits]

def make_skill_agent(k=3):
    # Same NB0 harness; per question we retrieve from Qdrant and inject (C <- S).
    def agent_fn(question):
        return make_agent(extra=format_skills(retrieve(question, k)))(question)
    return agent_fn

print("indexed", index_skills(candidate_skills), "skills into Qdrant ->", COLLECTION)
demo_q = "Which product generated the most completed revenue?"
print("retrieved for:", demo_q)
for s in retrieve(demo_q, k=3):
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

Re-index **only the curated** skills into Qdrant, then run with **retrieval** and
compare the three regimes.
"""),
    code(r"""
index_skills(curated)     # the vector DB now holds only the gated skills
METER.reset()
acc_curated = evaluate(make_skill_agent(k=3), split="test")["accuracy"]
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
- Retrieval from a real **vector DB (Qdrant)** means each prompt sees only the few
  skills it needs - the same RAG pattern you'd ship in production.

### The gap this leaves (-> NB4)
We curated with a *one-shot* rubric and a hand-set threshold. We never **optimized**
the library: no learning rate, no held-out **validation gate**, no momentum. NB4
(**SkillOpt**) treats the skill document as a trainable parameter and tunes it the
way you'd train a neural net - so improvement is systematic, not lucky.

### Exercise
1. Lower the gate to "average >= 1" instead of "min >= 1". Do more skills slip
   through, and does test accuracy fall? You just felt the 25%-degrade trap.
2. Sweep `k` in `make_skill_agent(k=...)` from 1 to 6. More retrieved skills isn't
   always better - find the sweet spot, and watch the cost meter climb with `k`.
3. Open the Qdrant dashboard and inspect the `workshop_skills` collection. Add a
   payload field (e.g. the skill's difficulty) and filter retrieval on it.
"""),
]


# ============================================================================
# NB4 -- SkillOpt: train the skill document like a neural net
# ============================================================================

NB4 = [
    md(r"""
# NB4 - SkillOpt: Train the Skill Document Like a Neural Net

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

NB3 built a skill *library*, but it never **optimized** it: we curated with a
one-shot rubric and a hand-set threshold, then hoped. NB4 turns that into a proper
**training loop**. The mental model of the whole workshop, made literal:

> **Reflection is the gradient, the skill document is the parameter vector, and
> your eval set is the loss.**

So we run **gradient descent in text space**:

| Neural net | SkillOpt (this notebook) |
|---|---|
| parameter vector theta | the **skill document** we inject (C <- S) |
| loss | **error on a held-out validation split** |
| gradient | a **reflection** that proposes a skill *edit* from a failure |
| learning rate | how many edits we try per step |
| **validation gate** | accept an edit only if it does **not** hurt val |
| momentum | the document **persists and compounds** across steps |

The **validation gate** is the star: it is exactly what stops the 25%-degrade
trap from NB3. We log the run to **Weights & Biases**, like a real fine-tune -
except the only thing changing is text, and there is no GPU in sight.
"""),
    code(r"""
import sys, os, json, re
sys.path.insert(0, os.path.abspath(".."))
from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_agent,
    preflight, flush,
)
preflight("WANDB_API_KEY")    # OPENAI + LANGFUSE + W&B (see SETUP.md)
build_db()

# Data discipline: TEST stays held out for the final number. We split the 24
# TRAIN tasks into an optimization set (where we look for failures to learn from)
# and a small VALIDATION set (the gate that decides which edits we keep).
train = [t for t in load_tasks() if t["split"] == "train"]
val   = train[::4]                          # every 4th train task -> the gate
opt   = [t for t in train if t not in val]
print(f"opt set: {len(opt)} tasks   val set: {len(val)} tasks   (test held out)")

try:
    baseline_acc = json.load(open("../data/baseline_test.json"))["accuracy"]
except FileNotFoundError:
    baseline_acc = evaluate(make_agent(), split="test")["accuracy"]
print("NB1 baseline test accuracy:", round(baseline_acc, 3))
"""),
    md(r"""
## The parameter and the gradient

`theta` (our parameter) is just a **list of skills**, injected through the same
`extra=` slot NB2/NB3 used - the frozen agent never changes. We need three pieces:
a way to **measure** a skill document (forward pass + loss), and a way to turn a
failure into a **proposed edit** (the gradient).
"""),
    code(r"""
def format_skills(skills):
    if not skills:
        return ""
    lines = ["Relevant skills (reusable SQL patterns learned from past tasks):"]
    for s in skills:
        lines.append(f"- {s['name']}: USE WHEN {s['when_to_use']}. PATTERN: {s['pattern']}")
    return "\n".join(lines)

def evaluate_theta(theta, tasks):
    # One forward pass: return (accuracy, failures) for skill document `theta`.
    agent = make_agent(extra=format_skills(theta))
    fails, correct = [], 0
    for t in tasks:
        sql = agent(t["question"])
        if score_sql(sql, t["gold"]):
            correct += 1
        else:
            fails.append({"question": t["question"], "sql": sql, "gold": t["gold"]})
    return correct / len(tasks), fails

def val_accuracy(theta):
    return evaluate_theta(theta, val)[0]      # the validation "loss" (as accuracy)

PROPOSE_SYS = (
    "You improve a text-to-SQL agent by writing ONE reusable SKILL that would have "
    "prevented a specific failure. Return STRICT JSON with keys name, when_to_use, "
    "pattern. 'when_to_use' triggers a CLASS of questions (no specific values); "
    "'pattern' is the general SQL technique. Generalize - never mention the question."
)

def parse_json_obj(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def propose_skill(failure):                    # the "gradient": failure -> edit
    raw = llm([
        {"role": "system", "content": PROPOSE_SYS},
        {"role": "user", "content":
            "Schema:\n" + SCHEMA_TEXT +
            "\nQuestion: " + failure["question"] +
            "\n\nAgent's wrong SQL:\n" + failure["sql"] +
            "\n\nCorrect SQL:\n" + failure["gold"] +
            "\n\nWrite ONE general skill (JSON) that fixes this class of mistake."},
    ])
    d = parse_json_obj(raw)
    if not d or not all(k in d for k in ("name", "when_to_use", "pattern")):
        return None
    return {k: str(d[k]) for k in ("name", "when_to_use", "pattern")}
"""),
    md(r"""
## The optimizer: learning rate + validation gate + momentum

Each step is one round of "SGD":
1. **Forward pass** - run the current `theta` on the opt set; collect failures.
2. **Gradient** - turn up to `LR` failures into candidate skill edits.
3. **Validation gate** - apply each candidate and keep it **only if val accuracy
   does not drop**. A candidate that would pollute the prompt is rejected here,
   before it can ever hurt the held-out test number.
4. **Momentum** - accepted skills stay in `theta` and compound into the next step.

We also stash *every* proposal (kept or not) so we can run the no-gate ablation.
"""),
    code(r"""
import wandb

N_STEPS = 4          # optimization steps ("epochs")
LR      = 2          # candidate skill edits proposed per step (the learning rate)

run = wandb.init(project="rl-agents-workshop", name="nb4-skillopt",
                 config={"model": "gpt-4o-mini", "n_steps": N_STEPS, "lr": LR,
                         "n_opt": len(opt), "n_val": len(val)})
wandb.define_metric("step")
wandb.define_metric("train_acc", step_metric="step")
wandb.define_metric("val_acc", step_metric="step")

theta = []                    # the skill document (parameter) - starts empty
all_candidates = []           # every proposal, for the no-gate ablation later
curve = []
METER.reset()
for step in range(N_STEPS):
    train_acc, fails = evaluate_theta(theta, opt)        # forward pass
    vacc = val_accuracy(theta)
    curve.append((step, train_acc, vacc, len(theta)))
    wandb.log({"step": step, "train_acc": train_acc, "val_acc": vacc,
               "n_skills": len(theta)})
    print(f"step {step}: train={train_acc:.2f}  val={vacc:.2f}  skills={len(theta)}")
    if not fails:
        print("  no failures left on the opt set -> converged"); break
    for f in fails[:LR]:                                  # gradient: LR edits
        cand = propose_skill(f)
        if not cand:
            continue
        all_candidates.append(cand)
        trial_val = val_accuracy(theta + [cand])          # the validation gate
        if trial_val >= vacc:
            theta.append(cand); vacc = trial_val
            print(f"  KEEP   {cand['name']:<26} val -> {trial_val:.2f}")
        else:
            print(f"  REJECT {cand['name']:<26} val {trial_val:.2f} < {vacc:.2f} (would pollute)")

print(f"\nfinal skill document: {len(theta)} skills (from {len(all_candidates)} proposed)")
print(METER)
"""),
    code(r"""
import matplotlib.pyplot as plt
xs = [c[0] for c in curve]
plt.figure(figsize=(6, 4))
plt.plot(xs, [c[1] for c in curve], marker="o", label="train acc (opt set)")
plt.plot(xs, [c[2] for c in curve], marker="s", label="val acc (the gate)")
plt.axhline(baseline_acc, ls="--", color="gray", label="NB0 baseline (test)")
plt.xlabel("optimization step"); plt.ylabel("accuracy"); plt.ylim(0, 1)
plt.title("SkillOpt: training the skill document (weights frozen)")
plt.legend(); plt.show()
"""),
    md(r"""
## Ablation - turn the gate off and watch it pollute

The gate is the only thing separating "learning" from "accumulating junk". Drop
it - accept **every** proposed edit, the unvetted pool from NB3 - and compare the
held-out **test** number against the gated document.
"""),
    code(r"""
theta_ungated = all_candidates                # keep everything we ever proposed

METER.reset()
test_gated   = evaluate(make_agent(extra=format_skills(theta)),         split="test")["accuracy"]
test_ungated = evaluate(make_agent(extra=format_skills(theta_ungated)), split="test")["accuracy"]
print("baseline (no skills)        :", round(baseline_acc, 3))
print("SkillOpt + gate  (NB4)      :", round(test_gated, 3),   f"  [{len(theta)} skills]")
print("accept-everything (no gate) :", round(test_ungated, 3), f"  [{len(theta_ungated)} skills]")
print(METER)

wandb.summary["baseline_acc"]   = baseline_acc
wandb.summary["test_gated"]     = test_gated
wandb.summary["test_ungated"]   = test_ungated
wandb.summary["n_skills_final"] = len(theta)
wandb.finish()
"""),
    code(r"""
import matplotlib.pyplot as plt
labels = ["baseline\n(NB0)", "SkillOpt\n+ gate", "no gate\n(accept all)"]
vals = [baseline_acc, test_gated, test_ungated]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, vals, color=["gray", "seagreen", "indianred"])
plt.ylim(0, 1); plt.ylabel("test accuracy")
plt.title("The validation gate is what makes optimization safe")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.show()
flush()
"""),
    md(r"""
## Takeaways

- **Optimizing a prompt is training.** We ran SGD with a parameter (the skill
  document), a loss (validation error), a gradient (reflection), a learning rate,
  and momentum - no weights, no GPU.
- The **validation gate** is the load-bearing idea: it turns "self-improvement"
  from a hope into a guarantee that each accepted edit was *measured* not to hurt.
- Tracked in **W&B** like any training run, so the curve, the baseline, and the
  gated-vs-ungated result are all on one shareable run.

### The gap this leaves (-> NB5)
Our document is **flat** - one list, every skill equal, retrieved the same way.
Real libraries are **hierarchical** (broad strategies -> specific patterns), and
the best skills are often written by a **strong** model and *transferred* to a
weaker, cheaper one. That is NB5.

### Exercise
1. Shrink `val` to 2 tasks. Does the gate get noisier (more bad keeps)? You just
   felt validation-set variance - the same bias/variance trade-off as in real ML.
2. Raise `LR` to 4. More edits per step costs more API calls (watch the meter) -
   does test accuracy actually improve, or do you just overfit the opt set?
3. Make the gate strict (`trial_val > vacc`). Fewer skills survive - compare the
   final test number and the skill count.
"""),
]


# ============================================================================
# NB5 -- Hierarchical skill library + strong->weak transfer
# ============================================================================

NB5 = [
    md(r"""
# NB5 - Hierarchical Skills and Strong -> Weak Transfer

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

NB4 trained a **flat** skill document: one list, every skill equal, all retrieved
the same way. Two ideas from the skill-optimization literature (SkillX, the
hierarchical-library work) make this scale:

1. **Hierarchy.** Organize skills as **strategies -> tactics**. Retrieve the right
   *strategy* for a question first, then pull only its *tactics*. This is
   coarse-to-fine RAG: fewer, more relevant tokens per prompt.
2. **Strong -> weak transfer.** The *author* of a skill matters. A **strong**
   model (gpt-4o) writes sharper, more general skills than a **weak** one
   (gpt-4o-mini). Inject the strong model's skills into the **weak** agent and the
   weak agent gets better - capability **transferred as text**, no fine-tuning.
   This is how a cheap production model can punch above its weight.

> **Cost note:** this notebook calls **gpt-4o** as the teacher, so it costs a bit
> more than the others (still cents). The *consumer* stays gpt-4o-mini.
"""),
    code(r"""
import sys, os, json, re
sys.path.insert(0, os.path.abspath(".."))
from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, embed, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_agent,
    preflight, flush,
)
preflight("QDRANT_URL", "QDRANT_API_KEY")    # + OPENAI + LANGFUSE (see SETUP.md)
build_db()

STRONG_MODEL = "gpt-4o"                                   # the teacher (authors skills)
WEAK_MODEL   = os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini")  # the consumer

try:
    baseline_acc = json.load(open("../data/baseline_test.json"))["accuracy"]
except FileNotFoundError:
    baseline_acc = evaluate(make_agent(model=WEAK_MODEL), split="test")["accuracy"]
print("weak baseline (no skills) test accuracy:", round(baseline_acc, 3))
"""),
    md(r"""
## 1. Author skills with a strong vs a weak teacher

We have **gold SQL** for the train tasks, so a teacher can write a skill straight
from each `(question, gold)` pair - no need to run an agent first. We ask each
teacher for a skill plus the **family** (strategy) it belongs to, which gives us
the hierarchy for free. We author from **medium/hard** train tasks (the easy ones
aren't worth a skill).
"""),
    code(r"""
AUTHOR_SYS = (
    "You write ONE reusable SKILL for a text-to-SQL agent from a solved example. "
    "Return STRICT JSON with keys: family, name, when_to_use, pattern. "
    "'family' is a short strategy category shared by similar skills "
    "(e.g. completed_revenue, set_difference, grouped_aggregation, date_bucketing). "
    "'when_to_use' triggers a CLASS of questions (no specific values); "
    "'pattern' is the general SQL technique. Generalize - never mention the question."
)

def parse_json_obj(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def author_skill(task, model):
    raw = llm([
        {"role": "system", "content": AUTHOR_SYS},
        {"role": "user", "content":
            "Schema:\n" + SCHEMA_TEXT +
            "\nSolved question: " + task["question"] +
            "\nCorrect SQL:\n" + task["gold"] +
            "\n\nReturn the skill as JSON."},
    ], model=model)
    d = parse_json_obj(raw)
    keys = ("family", "name", "when_to_use", "pattern")
    if not d or not all(k in d for k in keys):
        return None
    return {k: str(d[k]) for k in keys}

rich = [t for t in load_tasks()
        if t["split"] == "train" and t["level"] in ("medium", "hard")][:12]

METER.reset()
strong_skills = [s for s in (author_skill(t, STRONG_MODEL) for t in rich) if s]
weak_skills   = [s for s in (author_skill(t, WEAK_MODEL)   for t in rich) if s]
print(f"strong ({STRONG_MODEL}) authored {len(strong_skills)} skills")
print(f"weak   ({WEAK_MODEL}) authored {len(weak_skills)} skills")
print(METER)
"""),
    md(r"""
## 2. See the hierarchy the strong teacher produced

Grouping the strong model's skills by `family` gives a **strategies -> tactics**
tree. This structure is what we'll retrieve over: pick a strategy, then its tactics.
"""),
    code(r"""
from collections import defaultdict
def as_tree(skills):
    fams = defaultdict(list)
    for s in skills:
        fams[s["family"]].append(s)
    return fams

for fam, members in as_tree(strong_skills).items():
    print(f"STRATEGY  {fam}")
    for m in members:
        print(f"    - {m['name']}: USE WHEN {m['when_to_use'][:70]}")
"""),
    md(r"""
## 3. Index the hierarchy in Qdrant and retrieve coarse-to-fine

We store two kinds of points in one collection: **strategy** points (one per
family) and **tactic** points (one per skill), tagged with `level` and `family`.
Retrieval is two-stage, using Qdrant payload **filters**:
1. find the nearest **strategy** to the question, then
2. find the top tactics **within that family**.
"""),
    code(r"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
)

COLLECTION = "workshop_skills_hier"
EMBED_DIM = 1536                                   # text-embedding-3-small

qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

def index_hier(skills, collection=COLLECTION):
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
    points, pid = [], 0
    for fam, members in as_tree(skills).items():
        desc = "covers: " + ", ".join(m["name"] for m in members)
        points.append(PointStruct(id=pid, vector=embed(f"{fam}. {desc}"),
            payload={"level": "strategy", "family": fam, "description": desc}))
        pid += 1
        vecs = embed([f"{m['name']}. USE WHEN {m['when_to_use']}. {m['pattern']}"
                      for m in members])
        for m, v in zip(members, vecs):
            points.append(PointStruct(id=pid, vector=v,
                payload={"level": "tactic", **m})); pid += 1
    qdrant.upsert(collection, points=points)
    return qdrant.count(collection).count

def _only(level, family=None):
    must = [FieldCondition(key="level", match=MatchValue(value=level))]
    if family is not None:
        must.append(FieldCondition(key="family", match=MatchValue(value=family)))
    return Filter(must=must)

def retrieve_hier(question, k=3, collection=COLLECTION):
    qv = embed(question)
    strat = qdrant.query_points(collection, query=qv, limit=1,
                                query_filter=_only("strategy")).points
    if not strat:
        return []
    fam = strat[0].payload["family"]
    tactics = qdrant.query_points(collection, query=qv, limit=k,
                                  query_filter=_only("tactic", fam)).points
    return strat + tactics

def format_hier(points):
    if not points:
        return ""
    lines = ["Relevant skill hierarchy (apply the strategy, then its tactics):"]
    for p in points:
        d = p.payload
        if d["level"] == "strategy":
            lines.append(f"STRATEGY {d['family']} - {d['description']}")
        else:
            lines.append(f"  - {d['name']}: USE WHEN {d['when_to_use']}. PATTERN: {d['pattern']}")
    return "\n".join(lines)

print("indexed", index_hier(strong_skills), "points (strategies + tactics)")
demo_q = "Which customer has the highest total completed revenue?"
print("\nretrieved for:", demo_q)
print(format_hier(retrieve_hier(demo_q, k=2)))
"""),
    md(r"""
## 4. The transfer experiment

Same **weak** consumer agent (gpt-4o-mini), same hierarchical retrieval - we only
change **who authored the skills**. If strong -> weak transfer is real, the weak
agent should improve *more* with the strong teacher's skills than with its own.
"""),
    code(r"""
def make_consumer(k=3):
    # The weak agent, augmented per-question with retrieved hierarchical skills.
    def agent_fn(question):
        return make_agent(model=WEAK_MODEL,
                          extra=format_hier(retrieve_hier(question, k)))(question)
    return agent_fn

METER.reset()
index_hier(strong_skills)
acc_strong = evaluate(make_consumer(k=3), split="test")["accuracy"]
index_hier(weak_skills)
acc_weak = evaluate(make_consumer(k=3), split="test")["accuracy"]
print("weak baseline (no skills)        :", round(baseline_acc, 3))
print("weak + WEAK-authored skills      :", round(acc_weak, 3))
print("weak + STRONG-authored skills    :", round(acc_strong, 3), " <- transfer")
print(METER)
"""),
    code(r"""
import matplotlib.pyplot as plt
labels = ["weak\nbaseline", "weak +\nweak skills", "weak +\nstrong skills"]
vals = [baseline_acc, acc_weak, acc_strong]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, vals, color=["gray", "steelblue", "seagreen"])
plt.ylim(0, 1); plt.ylabel("test accuracy (weak consumer)")
plt.title("Strong -> weak transfer: better authorship lifts the cheap model")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.show()
flush()
"""),
    md(r"""
## Takeaways

- A **hierarchical** library (strategy -> tactic) retrieves coarse-to-fine, so
  each prompt sees fewer, more on-target skills than a flat dump.
- **Authorship matters.** Skills written by a strong model **transfer** to a weak
  one and lift its accuracy - capability moved as *text*, with the frozen weak
  model still doing inference. This is the cheap-model-punching-up pattern.
- Qdrant payload **filters** turn one collection into a multi-level index - the
  same trick scales to real production skill libraries.

### The gap this leaves (-> NB6)
We still ran each stage **by hand**: author, index, evaluate. NB6 closes the loop -
an **autonomous** agent that evolves its own skill library across generations,
**audits** every change, and reports a final number. The capstone.

### Exercise
1. Sweep `k` (tactics retrieved) from 1 to 5. Where does more context start to
   hurt? Watch the cost meter climb with `k`.
2. Add a third teacher tier (e.g. `gpt-4.1-mini`) and compare the transfer curve.
3. Retrieve the **top-2 strategies** instead of 1 before pulling tactics. Does
   broadening the coarse step help the hard, multi-concept questions?
"""),
]


# ============================================================================
# NB6 -- Capstone: self-evolving, auditable agent (EvoSkill + ASG-SI)
# ============================================================================

NB6 = [
    md(r"""
# NB6 - Capstone: A Self-Evolving, Auditable Agent

**Workshop: Self-Evolving Agents by Optimizing the Harness (no GPU)**

This is the payoff. Every component of **H = (E, T, C, S, L, V)** is now in play,
and we let the agent **evolve itself** - no human in the per-step loop.

- **EvoSkill** (evolutionary skill search): each generation breeds a small
  **population** of skill-library variants by *mutating* the current best -
  **adding** a skill distilled from a failure, or **dropping** a weak one - then
  **selects** the fittest on a held-out validation split. This is the outer
  **execution loop (E)** evolved into an optimizer.
- **ASG-SI** (auditable skill generation + self-improvement): every proposed
  change is scored with a **decomposed verifiable reward** (`score_sql`, our V)
  and written to an **audit log** with provenance - which generation, which
  operator, the measured val delta, and the accept/reject decision. Nothing enters
  the library (state **S**) without a recorded, replayable justification.

The result is a frozen LLM that gets measurably better **and** can explain exactly
*why* it changed - self-improvement you could put in front of a reviewer.
"""),
    code(r"""
import sys, os, json, re
sys.path.insert(0, os.path.abspath(".."))
from workshop_utils import (
    build_db, load_tasks, run_sql, score_sql, evaluate,
    llm, METER, SCHEMA_TEXT, extract_sql, baseline_prompt, make_agent,
    preflight, flush,
)
preflight("WANDB_API_KEY")    # OPENAI + LANGFUSE + W&B (see SETUP.md)
build_db()

train = [t for t in load_tasks() if t["split"] == "train"]
val   = train[::4]                          # held-out validation = fitness signal
opt   = [t for t in train if t not in val]  # where we mine failures to mutate from

try:
    baseline_acc = json.load(open("../data/baseline_test.json"))["accuracy"]
except FileNotFoundError:
    baseline_acc = evaluate(make_agent(), split="test")["accuracy"]
print(f"opt={len(opt)}  val={len(val)}  (test held out)")
print("baseline test accuracy:", round(baseline_acc, 3))
"""),
    md(r"""
## The pieces: measure, mutate, audit

We reuse the SkillOpt machinery (forward pass + the reflection "gradient"), then
add the two evolutionary **mutation operators** and the **audit log** that makes
the run reviewable.
"""),
    code(r"""
def format_skills(skills):
    if not skills:
        return ""
    lines = ["Relevant skills (reusable SQL patterns the agent has learned):"]
    for s in skills:
        lines.append(f"- {s['name']}: USE WHEN {s['when_to_use']}. PATTERN: {s['pattern']}")
    return "\n".join(lines)

def evaluate_theta(theta, tasks):
    agent = make_agent(extra=format_skills(theta))
    fails, correct = [], 0
    for t in tasks:
        sql = agent(t["question"])
        if score_sql(sql, t["gold"]):
            correct += 1
        else:
            fails.append({"question": t["question"], "sql": sql, "gold": t["gold"]})
    return correct / len(tasks), fails

def val_fitness(theta):
    return evaluate_theta(theta, val)[0]

PROPOSE_SYS = (
    "You improve a text-to-SQL agent by writing ONE reusable SKILL that would have "
    "prevented a failure. Return STRICT JSON with keys name, when_to_use, pattern. "
    "Generalize to a CLASS of questions; never mention the specific question."
)

def parse_json_obj(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def propose_skill(failure):                    # mutation: ADD (gradient from a failure)
    raw = llm([
        {"role": "system", "content": PROPOSE_SYS},
        {"role": "user", "content":
            "Schema:\n" + SCHEMA_TEXT + "\nQuestion: " + failure["question"] +
            "\n\nWrong SQL:\n" + failure["sql"] + "\n\nCorrect SQL:\n" + failure["gold"] +
            "\n\nWrite ONE general skill (JSON)."},
    ])
    d = parse_json_obj(raw)
    if not d or not all(k in d for k in ("name", "when_to_use", "pattern")):
        return None
    return {k: str(d[k]) for k in ("name", "when_to_use", "pattern")}
"""),
    md(r"""
## The outer loop: a (1 + lambda) evolutionary strategy

Each **generation**:
1. Measure the current best library (`parent`) - its val **fitness** and its
   failures on the opt set.
2. Breed `lambda` **offspring** by mutation: ADD a skill distilled from a failure,
   or DROP an existing skill (to fight bloat).
3. **Select** the fittest of {parent, offspring...} on validation (**elitism** -
   the parent survives unless beaten), and record every trial in the audit log.

The decomposed verifiable reward (`val_fitness`) is the selection pressure; the
audit log is the receipt.
"""),
    code(r"""
import wandb

N_GEN      = 4       # generations
LAMBDA_ADD = 2       # ADD-mutations bred per generation

run = wandb.init(project="rl-agents-workshop", name="nb6-evoskill",
                 config={"model": "gpt-4o-mini", "generations": N_GEN, "lambda": LAMBDA_ADD})
wandb.define_metric("generation")
wandb.define_metric("val_fitness", step_metric="generation")

parent = []                 # the evolving skill library (state S)
audit  = []                 # ASG-SI: every proposed change, scored and decided
curve  = []
METER.reset()

for gen in range(N_GEN):
    pfit, pfails = evaluate_theta(parent, opt)
    pval = val_fitness(parent)
    curve.append((gen, pval, len(parent)))
    wandb.log({"generation": gen, "val_fitness": pval, "library_size": len(parent)})
    print(f"gen {gen}: val_fitness={pval:.2f}  library={len(parent)} skills")

    # --- breed offspring by mutation ------------------------------------------
    offspring = []                                  # (operator, new_theta, note)
    for f in pfails[:LAMBDA_ADD]:
        c = propose_skill(f)
        if c:
            offspring.append(("ADD", parent + [c], c["name"]))
    if parent:                                      # DROP-mutation: try removing one
        offspring.append(("DROP", parent[:-1], parent[-1]["name"]))

    # --- select the fittest (elitism: parent must be beaten) -------------------
    best_fit, best_theta, best_op, best_note = pval, parent, "ELITE", "-"
    for op, cand_theta, note in offspring:
        f = val_fitness(cand_theta)
        accepted = f > best_fit
        audit.append({"gen": gen, "op": op, "skill": note,
                      "val_fitness": round(f, 3), "decision": "accept" if accepted else "reject"})
        print(f"    {op:<4} {note:<26} val={f:.2f}  -> {'ACCEPT' if accepted else 'reject'}")
        if accepted:
            best_fit, best_theta, best_op, best_note = f, cand_theta, op, note
    parent = best_theta

print(f"\nevolved library: {len(parent)} skills, val_fitness={val_fitness(parent):.2f}")
print(METER)
"""),
    code(r"""
import matplotlib.pyplot as plt
xs = [c[0] for c in curve]
plt.figure(figsize=(6, 4))
plt.plot(xs, [c[1] for c in curve], marker="o", color="seagreen", label="val fitness")
plt.axhline(baseline_acc, ls="--", color="gray", label="NB0 baseline (test)")
plt.xlabel("generation"); plt.ylabel("validation fitness"); plt.ylim(0, 1)
plt.title("EvoSkill: the library evolves itself (weights frozen)")
plt.legend(); plt.show()
"""),
    md(r"""
## The audit trail (ASG-SI)

Self-improvement you can review: every change the agent *considered*, the
**measured** validation reward, and the accept/reject decision. Nothing entered
the library without this receipt.
"""),
    code(r"""
print(f"{'gen':>3}  {'operator':<6} {'skill':<26} {'val':>5}  decision")
print("-" * 60)
for a in audit:
    print(f"{a['gen']:>3}  {a['op']:<6} {a['skill']:<26} {a['val_fitness']:>5}  {a['decision']}")

print("\nFinal library (provenance = the audit rows above that were accepted):")
for s in parent:
    print(f"  - {s['name']}: USE WHEN {s['when_to_use'][:64]}")
"""),
    md(r"""
## The capstone number

Report the evolved library on the **held-out test split** - the number we never
optimized against - and place it next to the NB0 baseline.
"""),
    code(r"""
METER.reset()
final_test = evaluate(make_agent(extra=format_skills(parent)), split="test")
acc = final_test["accuracy"]
print("baseline (NB0, no skills)      :", round(baseline_acc, 3))
print("self-evolved agent (NB6)       :", round(acc, 3))
print("by level:", {k: round(v["acc"], 2) for k, v in final_test["by_level"].items()})
print(METER)

wandb.summary["baseline_acc"]   = baseline_acc
wandb.summary["final_test_acc"] = acc
wandb.summary["library_size"]   = len(parent)
wandb.summary["audit_events"]   = len(audit)
wandb.finish()
flush()
"""),
    md(r"""
## Takeaways - the whole thesis, realized

- We built an agent (NB0), measured it (NB1, **V**), gave it memory (NB2, **S**),
  structured that into a gated skill library (NB3), **trained** it like a net
  (NB4), made it **hierarchical and transferable** (NB5), and finally let it
  **evolve itself with an audit trail** (NB6) - all with the **weights frozen**.
- **Reflection was the gradient, the skill document was the parameter vector, and
  the eval set was the loss.** Self-evolution = optimizing the **harness**, not the
  brain. No GPU, on a laptop, for cents.
- **Auditable** beats merely "better": the ASG-SI log means every gain is
  traceable to a measured reward and a recorded decision - the difference between a
  demo and something you would actually ship.

### Where to take it next
1. Replace the `(1 + lambda)` strategy with a real **population** (crossover
   between two libraries). Does diversity find skills a greedy search misses?
2. Add a **regression test**: re-score accepted skills each generation and DROP any
   whose contribution has decayed. (Skills can go stale as the library changes.)
3. Swap the toy DB for your own schema and tasks. The harness is the product - the
   database is just where you point it.
"""),
]


if __name__ == "__main__":
    build(os.path.join(NB_DIR, "NB0_build_your_first_agent.ipynb"), NB0)
    build(os.path.join(NB_DIR, "NB1_eval_interface_and_baseline.ipynb"), NB1)
    build(os.path.join(NB_DIR, "NB2_reflexion.ipynb"), NB2)
    build(os.path.join(NB_DIR, "NB3_skill_lifecycle.ipynb"), NB3)
    build(os.path.join(NB_DIR, "NB4_skillopt.ipynb"), NB4)
    build(os.path.join(NB_DIR, "NB5_hierarchical_transfer.ipynb"), NB5)
    build(os.path.join(NB_DIR, "NB6_capstone_self_evolving.ipynb"), NB6)
    print("done")
