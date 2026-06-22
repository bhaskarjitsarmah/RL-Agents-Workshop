# Workshop Architecture: Self-Evolving Agents by Optimizing the Harness

> **The one thesis:** an agent = a **frozen brain** + an **evolving harness**.
> We never touch the model weights (that needs a GPU). We make the agent *learn*
> by optimizing everything *around* the brain - its prompt, tools, memory, and
> control loop - in **text space**, on a laptop.
>
> **Reflection is the gradient · the skill document is the parameter vector · the
> eval set is the loss.**

This document is the map. It connects all seven notebooks (NB0 -> NB6) into one
picture and shows how each one builds or evolves a piece of the agent.

> 🖼️ **Slide-ready exports** (SVG + PNG) of every diagram below live in
> [`docs/diagrams/`](docs/diagrams/).

---

## 1. The anatomy of the agent: `H = (E, T, C, S, L, V)`

Every agent in this workshop is the same frozen LLM wrapped in a harness with six
named parts. The harness is the product; the brain is a fixed component.

```mermaid
flowchart TB
    User["NL question<br/>(How many customers in Mumbai?)"]

    subgraph FROZEN["BRAIN — frozen, never fine-tuned, no GPU"]
        LLM["LLM weights<br/>gpt-4o-mini"]
    end

    subgraph HARNESS["HARNESS  H = (E, T, C, S, L, V) — everything we evolve"]
        C["C · Context manager<br/>schema + retrieved skills -> the prompt"]
        T["T · Tool registry<br/>run_sql: execute against the DB"]
        E["E · Execution loop<br/>generate -> run -> observe -> retry"]
        S["S · State store<br/>skill / memory library"]
        L["L · Lifecycle hooks<br/>validation gates: what to keep / when to stop"]
        V["V · Evaluation interface<br/>score_sql: execution-match reward"]
    end

    User --> C
    C --> LLM
    LLM -->|"candidate SQL"| E
    E --> T
    T -->|"rows / error"| E
    E -->|"answer"| V
    V -->|"reward signal"| S
    S -->|"retrieve relevant skills"| C
    L -.->|"gate updates to"| S

    style FROZEN fill:#eee,stroke:#888
    style HARNESS fill:#f3f9f3,stroke:#5a5
```

**Read it as a loop:** the brain proposes SQL, the execution loop runs the tool and
observes the result, the evaluation interface turns the result into a reward, the
reward updates the skill library, and the library feeds back into the context the
brain sees next time. The lifecycle hooks decide which updates are allowed in.
**The weights never move - the loop does the learning.**

---

## 2. The notebook journey: build it, then evolve it

NB0 *builds* the first agent (E, T, C). Every notebook after that *evolves* one
part of the harness. The accuracy on a held-out test set is the scoreboard.

```mermaid
flowchart LR
    NB0["NB0 · Build the agent<br/>E + T + C<br/><i>S and V still empty</i>"]
    NB1["NB1 · Eval interface<br/><b>V</b> = score_sql<br/><i>you can't improve<br/>what you can't measure</i>"]
    NB2["NB2 · Reflexion<br/><b>S + C</b><br/>free-text lessons<br/><i>reflection = gradient</i>"]
    NB3["NB3 · Skill lifecycle<br/>structured <b>S</b> + gate<br/>generate->extract->consume<br/><i>the 25%-degrade trap</i>"]
    NB4["NB4 · SkillOpt<br/>train <b>S</b> like a net<br/>LR + val-gate + momentum"]
    NB5["NB5 · Hierarchy + transfer<br/><b>S</b> = strategy->tactic<br/>strong->weak teaching"]
    NB6["NB6 · Capstone EvoSkill<br/>autonomous <b>E</b> + ASG-SI audit<br/><i>self-evolving + reviewable</i>"]

    NB0 --> NB1 --> NB2 --> NB3 --> NB4 --> NB5 --> NB6
```

| NB | Title | Harness part built / evolved | The new idea |
|---|---|---|---|
| **NB0** | Build your first agent | **E, T, C** (S, V named but empty) | agent = brain + harness; the loop handles *crashes* for free |
| **NB1** | The eval interface | **V** | an objective, replayable reward (execution match) - the prerequisite for all learning |
| **NB2** | Reflexion | **S + C** | learn from mistakes as natural-language lessons; *reflection is the gradient* |
| **NB3** | The skill lifecycle | structured **S** + **L** | skills are *structured & retrieved*; an unvetted pool degrades (~25%) -> you need a gate |
| **NB4** | SkillOpt | optimize **S** | the skill document is a *trainable parameter*: learning rate, **validation gate**, momentum |
| **NB5** | Hierarchy + transfer | **S** structure | strategy->tactic retrieval; a strong model's skills *transfer* to a weak one |
| **NB6** | Capstone | evolved **E** (+ S, L, V) | an autonomous evolutionary loop with an **audit trail** - self-improvement you could ship |

---

## 3. The engine: the self-evolution loop (RL without gradients)

This is the heart of the workshop. It is reinforcement learning where the
**policy is the harness** and the learning signal is **verbal/evolutionary
feedback** instead of backprop. NB2 runs one turn of it by hand; NB4 makes it an
optimizer; NB6 makes it autonomous.

```mermaid
flowchart TB
    A["1 · GENERATE<br/>run the agent on tasks<br/>(collect trajectories)"]
    B["2 · REWARD (V)<br/>score_sql execution-match<br/>= decomposed verifiable reward"]
    R["3 · REFLECT = the GRADIENT<br/>compare wrong SQL vs gold,<br/>distill a general lesson/skill"]
    D["4 · PROPOSE EDIT<br/>add / drop / distill a skill<br/>(Δ to the parameter)"]
    G{"5 · VALIDATION GATE (L)<br/>does held-out val accuracy hold?"}
    H["6 · UPDATE LIBRARY (S)<br/>versioned + audited skill store"]
    I["7 · INJECT into CONTEXT (C)<br/>retrieve relevant skills (Qdrant)"]

    A --> B --> R --> D --> G
    G -- "keep (no regression)" --> H
    G -- "reject (would pollute)" --> A
    H --> I --> A

    style G fill:#fff3cd,stroke:#d4a017
    style B fill:#e8f0fe,stroke:#4285f4
    style H fill:#f3f9f3,stroke:#5a5
```

**The ML analogy, made literal:**

| Neural-net training | This workshop (text space) | Where |
|---|---|---|
| parameter vector θ | the **skill document** (list of skills in the prompt) | NB2-NB6 |
| loss | **error on a held-out eval split** | NB1 (V) |
| gradient | a **reflection** that proposes a skill edit | NB2 |
| learning rate | how many edits we accept per step | NB4 |
| regularization / early stop | the **validation gate** that blocks pollution | NB3, NB4 |
| momentum | the library **persists and compounds** | NB4 |
| population / evolution | EvoSkill mutate + select across generations | NB6 |
| training run dashboard | **Weights & Biases** curves | NB2, NB4, NB6 |

The gate (step 5) is the load-bearing idea: it is the difference between *learning*
and *accumulating junk*, and it is exactly what defuses NB3's 25%-degrade trap.

---

## 4. The real-world tool stack

The workshop runs the way an AI engineer actually works - a live observability +
experiment-tracking + vector-DB stack, not in-memory stubs. One wrapper (`llm()`)
is the choke point everything is instrumented through.

```mermaid
flowchart LR
    subgraph Code["workshop_utils (the plumbing)"]
        LLMW["llm() wrapper<br/>+ cost meter"]
        DBW["db.py<br/>toy shop DB + score_sql (V)"]
        AG["agents.py<br/>make_agent(extra=...)"]
    end

    subgraph Stack["Real AI-engineering stack"]
        OAI["OpenAI<br/>the brain · ALL notebooks"]
        LF["Langfuse<br/>trace every llm() call · ALL"]
        WB["Weights & Biases<br/>learning curves · NB2, NB4, NB6"]
        QD["Qdrant Cloud<br/>skill vector DB · NB3, NB5, NB6"]
    end

    LLMW --> OAI
    LLMW -."every call traced".-> LF
    AG --> LLMW
    AG -->|"retrieve skills"| QD
    AG -->|"reward via"| DBW
    AG -."curves logged".-> WB
```

| Tool | Role | Used in |
|---|---|---|
| **OpenAI** | the frozen brain (`gpt-4o-mini`; `gpt-4o` as teacher in NB5) | all |
| **Langfuse** | traces every `llm()` call = "collect trajectories" | all |
| **Weights & Biases** | logs the agent's "training" curves | NB2, NB4, NB6 |
| **Qdrant Cloud** | managed vector DB for skill retrieval (RAG) | NB3, NB5, NB6 |

---

## 5. The task spine: text-to-SQL (why it works)

Every notebook uses one task: translate a natural-language question into SQLite
over a toy "shop" DB (`customers -> orders -> order_items -> products`).

```mermaid
erDiagram
    customers ||--o{ orders : places
    orders ||--o{ order_items : contains
    products ||--o{ order_items : "appears in"
    customers {
        int customer_id PK
        text name
        text city
        text segment
    }
    products {
        int product_id PK
        text name
        text category
        real price
    }
    orders {
        int order_id PK
        int customer_id FK
        text order_date
        text status
    }
    order_items {
        int item_id PK
        int order_id FK
        int product_id FK
        int quantity
    }
```

Text-to-SQL is the spine because it is **auto-scorable** (execute predicted vs gold
SQL and compare result sets - no LLM judge), **skills compound** (join patterns,
the `status='completed'` revenue rule, set-difference idioms are obviously
reusable), and it's free, fast, and offline. That auto-scorability is what makes
`V` an *objective* reward, which is what makes the whole self-evolution loop possible.

---

## TL;DR for the room

1. **Agent = frozen brain + harness.** We optimize the harness, never the weights.
2. **`H = (E, T, C, S, L, V)`** names the parts; NB0 builds E/T/C, the rest evolve S via V.
3. **The loop is RL in text:** reflection is the gradient, the skill doc is the
   parameter, the eval set is the loss - and the **validation gate** keeps it honest.
4. **It's auditable and cheap:** every gain traces to a measured reward (NB6), and a
   full run costs cents on a laptop. No GPU, ever.
