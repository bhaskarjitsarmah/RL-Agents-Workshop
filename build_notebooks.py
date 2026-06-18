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
)

DB = build_db()          # deterministic rebuild; same data on every machine
print("Database ready at:", DB)
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
)
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


if __name__ == "__main__":
    build(os.path.join(NB_DIR, "NB0_build_your_first_agent.ipynb"), NB0)
    build(os.path.join(NB_DIR, "NB1_eval_interface_and_baseline.ipynb"), NB1)
    build(os.path.join(NB_DIR, "NB2_reflexion.ipynb"), NB2)
    print("done")
