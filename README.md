# Self-Evolving Agents by Optimizing the Harness (No GPU)

A full-day, hands-on workshop. We build agents that **learn and improve without
fine-tuning the model** - by evolving the *harness* around a frozen LLM:
its prompt, memory, examples, tools, and - the headline - its **skill library**.

> **Thesis:** *Reflection is the gradient, the skill document is the parameter
> vector, and your eval set is the loss.* You can "train" an agent in text space,
> on a laptop, with no GPU.

## The framing

We use the agent-harness formalism **H = (E, T, C, S, L, V)**. In **NB0** we
*build* the first agent - the initial execution loop (E), tool (T) and context
(C). Every notebook after that **evolves** one part of the harness:

| | Component | First built | What we evolve | Notebook |
|---|---|---|---|---|
| E | Execution loop | NB0 | the self-evolution outer loop | NB6 |
| T | Tool registry | NB0 | self-built, hierarchical skill library | NB5 |
| C | Context manager | NB0 | reflections -> distilled skills | NB2, NB3 |
| S | State store | - | persisted, versioned, audited skills | NB2, NB6 |
| L | Lifecycle hooks | NB0 | validation gates around skill promotion | NB4 |
| V | Evaluation interface | - | the reward signal (prerequisite for all) | NB1 |

This is **Reinforcement Learning for LLM agents**, but the policy is the harness
and the learning signal is verbal/evolutionary feedback instead of gradients.

## Module roadmap

| # | Module | Status |
|---|---|---|
| M0 | Harness anatomy & the thesis (slides) | - |
| **NB0** | **Build your first agent (and meet the harness)** | **built** |
| **NB1** | **The eval interface (V) + measuring your agent** | **built** |
| **NB2** | **Reflexion: reflection is the gradient** | **built** |
| **NB3** | **The skill lifecycle (generate -> extract -> consume) + the 25%-degrade trap** | **built** |
| **NB4** | **SkillOpt: train the skill document like a neural net** | **built** |
| **NB5** | **Hierarchical skill library + strong->weak transfer** | **built** |
| **NB6** | **Capstone: self-evolving, auditable agent (EvoSkill + ASG-SI)** | **built** |

Backing papers: SkillOpt, the MSFT/Fudan skill-lifecycle study, SkillX, EvoSkill,
ASG-SI (see workshop slides).

## Real-world tool stack (required)

This workshop is run the way an AI engineer actually works - with a live
observability + experiment-tracking + vector-DB stack, not toy in-memory stubs.
**Create these four free accounts and fill your `.env` _before_ the session**
(full walkthrough in **[SETUP.md](SETUP.md)**):

| Tool | Role in the workshop | Used in |
|---|---|---|
| **OpenAI** | the model ("brain") | all |
| **Langfuse** | LLM observability / tracing - every `llm()` call is traced | all |
| **Weights & Biases** | experiment tracking - the agent's "training" curves | NB2, NB4 |
| **Qdrant Cloud** | managed vector DB for skill / memory retrieval | NB3, NB5 |
| **LangGraph** | agent orchestration framework (lib, no account) | later refactor |

The notebooks **fail fast** (`preflight()`) if any required key is missing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys per SETUP.md
python -c "from workshop_utils import preflight; preflight()"   # verify
```

Any OpenAI-compatible endpoint works - set `OPENAI_BASE_URL` in `.env` to point at
Azure, a local vLLM/Ollama server, or a corporate proxy. Default model is
`gpt-4o-mini` (cheap + fast); change `WORKSHOP_MODEL` to swap.

## Running the notebooks

The notebooks live in `notebooks/` and are meant to be run **top-to-bottom, in
order** (NB0 -> NB1 -> ...). Each one's first cell calls `preflight()` and will
stop immediately with a clear message if a required key is missing. Pick either
front-end:

**VS Code**

1. Open the repo, then open `notebooks/NB0_build_your_first_agent.ipynb`.
2. Click **Select Kernel** (top-right) -> **Python Environments** -> choose the
   interpreter at **`.venv`** (`.venv\Scripts\python.exe` on Windows). If `.venv`
   isn't listed, reload the window (Ctrl/Cmd+Shift+P -> "Developer: Reload Window").
3. **Run All**, or run cells one at a time with Shift+Enter.

**Jupyter Lab**

```bash
jupyter lab notebooks/      # uses the .venv you activated above
```

Either way, the setup cell should print `preflight OK ->...` and
`Database ready at: ...`. Once that passes, the rest of the notebook just works.

## The task: text-to-SQL

Every notebook uses one spine: translate a natural-language question into a
SQLite query over a small toy "shop" database. Why text-to-SQL?

- **Auto-scorable** - we execute predicted vs. gold SQL and compare result sets
  (no LLM judge needed). This is the reward signal that makes evolution possible.
- **Skills compound** - schema cheatsheets, join patterns, and pitfall notes are
  obviously reusable, so the "improvement curve" story is strong.
- Free, fast, offline, and familiar to engineers.

40 tasks (`easy | medium | hard`), split 24 train / 16 test. We optimize on
train and report on test - never the other way around.

## Cost

Harness optimization is API-call-heavy, so every notebook prints a **cost meter**
(calls / tokens / approximate USD). On `gpt-4o-mini` a full run of NB0+NB1+NB2 is
a few cents. Tune `max_repairs` (the agent's retry loop) and the train-set size
to control spend.

## Repo layout

```
RL-Agents-Workshop/
  workshop_utils/        shared, reusable plumbing
    llm.py               llm() wrapper + cost meter + Langfuse tracing + preflight()
    db.py                toy DB builder + execution-match scorer (the reward V)
    tasks.py             40 NL -> gold SQL tasks (train/test)
    evaluate.py          the eval harness (component V)
    agents.py            extract_sql, baseline prompt, and make_agent (the looped agent)
  data/
    shop.db              generated deterministically (safe to delete/rebuild)
  notebooks/
    NB0_build_your_first_agent.ipynb
    NB1_eval_interface_and_baseline.ipynb
    NB2_reflexion.ipynb
    NB3_skill_lifecycle.ipynb
    NB4_skillopt.ipynb
    NB5_hierarchical_transfer.ipynb
    NB6_capstone_self_evolving.ipynb
  build_notebooks.py     regenerates the notebooks from plain-text cell defs
  requirements.txt
  .env.example
  SETUP.md               pre-work: accounts + keys to set up before the workshop
```

To regenerate notebooks after editing `build_notebooks.py`:

```bash
python build_notebooks.py
```
