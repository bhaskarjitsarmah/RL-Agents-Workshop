# Pre-work: set up your toolchain BEFORE the workshop

This is a hands-on, full-day workshop run the way a real AI engineer works - with
a live observability + experiment-tracking + vector-DB stack. **Please create
these four accounts and put the keys in your `.env` before the session.** All are
free tiers; doing this in advance means we spend the day building, not signing up.

> The notebooks **fail fast** with a clear message if a key is missing - so if
> your `.env` is complete, you're ready.

## 0. Clone + install

```bash
git clone https://github.com/bhaskarjitsarmah/RL-Agents-Workshop.git
cd RL-Agents-Workshop
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # Windows: copy .env.example .env
```

Then fill in `.env` using the four steps below.

## 1. OpenAI — the model ("brain")

1. Go to <https://platform.openai.com/api-keys> and create a key.
2. Put it in `.env` as `OPENAI_API_KEY`.
3. Add a few dollars of credit. A full run of every notebook is a few cents on
   `gpt-4o-mini`, but a key with **zero** balance returns errors.

*(Any OpenAI-compatible endpoint works too - set `OPENAI_BASE_URL` for Azure, a
local vLLM/Ollama server, or a corporate proxy.)*

## 2. Langfuse — observability / tracing

Every LLM call in the workshop is traced here; this is how we "collect
trajectories" the way a real team does.

1. Sign up at <https://cloud.langfuse.com> (pick the **US** or **EU** region and
   remember which - the URLs differ).
2. Create a project → **Settings → API Keys** → create keys.
3. Fill `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and set `LANGFUSE_BASE_URL`
   to your region (`https://us.cloud.langfuse.com` or `https://cloud.langfuse.com`).

## 3. Weights & Biases — experiment tracking

We log the agent's "training" curves (Reflexion in NB2, SkillOpt in NB4) as real
W&B runs.

1. Sign up at <https://wandb.ai>.
2. Copy your key from <https://wandb.ai/authorize>.
3. Put it in `.env` as `WANDB_API_KEY`.

## 4. Qdrant Cloud — vector database

The skill/memory library is retrieved from a managed vector DB.

1. Sign up at <https://cloud.qdrant.io> (free 1 GB cluster, no credit card).
2. Create a cluster; copy its **URL** into `QDRANT_URL` (include the `:6333` port).
3. Generate an **API key** for the cluster → `QDRANT_API_KEY`.

## Verify you're ready

```bash
python -c "from workshop_utils import preflight; preflight()"
```

You should see `preflight OK`. If a key is missing it will tell you exactly which
one. See you at the workshop!
