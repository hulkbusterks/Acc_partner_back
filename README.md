Accountability PoC
==================

Overview
--------
This repository contains the backend for an accountability study app.
It uses FastAPI and LangGraph to orchestrate RAG-based MCQ generation from
uploaded study material (PDFs, EPUBs, plain text). The system generates
questions grounded in the user's own content and tracks progress via
leaderboards and per-session audit trails.

### Vision — Guided Study Sessions (next milestone)

The end goal is an **AI-guided study timeline** where the system actively
manages the user's study flow:

1. **AI-generated study plan** — when a session starts, AI analyses the
   uploaded material and proposes a timeline with subtopics and estimated
   durations. The user reviews, adjusts, and finalises the plan.
2. **Timer-driven study mode** — the user sees only a running clock while
   studying. At random intervals the system pushes a "challenge notification"
   asking the user to answer a question.
3. **Adaptive question selection** — questions are chosen based on which
   subtopic the user should have reached according to the timeline. If the
   user indicates they haven't reached that subtopic yet, the timeline
   re-syncs and scoring adjusts accordingly (behind-schedule penalty,
   ahead-of-schedule bonus).
4. **Pause / resume** — sessions can be paused and resumed at any time;
   the countdown freezes while paused.
5. **Streak-based scoring** — consecutive correct answers build a streak
   multiplier. Breaking the streak resets it. The final score factors in
   streak, timeline adherence, and raw accuracy.
6. **Configurable frequency** — the user can set how often challenge
   notifications appear (e.g. every 5–15 minutes).

### Roadmap

| Phase | Feature | Est. Hours |
|-------|---------|------------|
| P1 | Pause / Resume sessions | 3–4 |
| P2 | Streak-based scoring | 2–3 |
| P3 | AI study plan & timeline generation | 8–12 |
| P4 | Adaptive question selection by milestone | 6–8 |
| P5 | Challenge notifications (polling) | 4–6 |
| P6 | Timeline adherence scoring | 3–4 |
| P7 | User notification preferences | 2–3 |
| **Total** | **Minimum viable guided sessions** | **28–40** |

Running locally (PoC mock mode)
--------------------------------
1. Create a virtualenv and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Set env vars (optional) — by default `LLM_PROVIDER=mock`, `VECTOR_PROVIDER=faiss`,
   `ORCHESTRATOR=mock`.

3. Start the app (when implemented):

```powershell
uvicorn server.main:app --reload
```

Notes
-----
- LangGraph v1 is targeted for the orchestrator; nodes are thin and call adapters.
- Do not add production credentials to the repo; provide them via `.env`.

Environment flags (developer)
-----------------------------

- `ORCHESTRATOR`: which orchestrator to use. Options: `mock` (default) or `langgraph`.
- `LANGGRAPH_GRAPH_PATH`: Path to a Python graph module (e.g. `langgraph_samples/rag_graph.py`) used for local LangGraph testing.
- `LLM_PROVIDER`: Optional; selects the LLM adapter implementation.
- `EMBEDDING_PROVIDER`: Optional (`auto` default). Options: `auto`, `adapter`, `hf`, `hash`.
- `EMBEDDING_MODEL`: Optional Hugging Face model id for `EMBEDDING_PROVIDER=hf` (default `sentence-transformers/all-MiniLM-L6-v2`).
- `ORCHESTRATOR_VERIFY`: If truthy, enables deterministic evidence verification in the orchestrator pipeline.
- `ORCHESTRATOR_MAX_RETRIES`: Optional integer (default `2`); max repair attempts when verification/quality checks fail.
- `ORCHESTRATOR_MAX_MCQS`: Optional integer (default `8`); upper bound on questions generated per orchestration call.
- `ORCHESTRATOR_MAX_TOPICS`: Optional integer (default `5`); upper bound on topic proposals.
- `ORCHESTRATOR_MIN_CONTEXT_WORDS`: Optional integer (default `4`); minimum context richness threshold for question quality checks.
- `ORCHESTRATOR_DRAFT_MODE`: Optional (`deterministic` default, `llm` supported) for question drafting strategy.
- `ORCHESTRATOR_MODEL`: Optional string tag to record selected model in graph metadata for traceability.
- `ORCHESTRATOR_ENABLE_MODERATION`: Optional boolean (default `false`); enables moderation checks on generated question payloads.
- `ORCHESTRATOR_RATE_LIMIT_PER_MIN`: Optional integer (default `30`); max `/sessions/{id}/generate_questions` calls per user per minute.
- `DATABASE_URL`: SQLAlchemy connection string (default `sqlite:///./acc_poc.db`).
- `ADMIN_TOKEN`: Required for `/leaderboard/reset`; if not set, reset endpoint is disabled.

Example `.env` for local deterministic orchestration:

```dotenv
ORCHESTRATOR=langgraph
LANGGRAPH_GRAPH_PATH=langgraph_samples/rag_graph.py
LLM_PROVIDER=groq
EMBEDDING_PROVIDER=hf
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ORCHESTRATOR_VERIFY=true
ORCHESTRATOR_MAX_RETRIES=2
ORCHESTRATOR_MAX_MCQS=8
ORCHESTRATOR_MAX_TOPICS=5
ORCHESTRATOR_MIN_CONTEXT_WORDS=4
ORCHESTRATOR_DRAFT_MODE=deterministic
ORCHESTRATOR_MODEL=default
ORCHESTRATOR_ENABLE_MODERATION=false
ORCHESTRATOR_RATE_LIMIT_PER_MIN=30
DATABASE_URL=sqlite:///./acc_poc.db
ADMIN_TOKEN=local-dev-admin-token
```

Running migrations (PowerShell)
------------------------------

```powershell
$env:PYTHONPATH = "$PWD"; .\.venv\Scripts\python.exe -m alembic upgrade head
```

Quick smoke and tests
---------------------

```powershell
$env:PYTHONPATH = "$PWD"; .\.venv\Scripts\python.exe scripts\smoke_test.py
$env:PYTHONPATH = "$PWD"; .\.venv\Scripts\python.exe -m pytest -q
```

Chunking defaults
-----------------

- Token-based chunk size default: `chunk_tokens=200`.
- Character fallback chunk size default: `chunk_size=1200`.

These defaults were chosen from local benchmark runs in `scripts/chunking_benchmark.py`.


API docs (Swagger)
------------------

The app exposes an OpenAPI UI at `/docs` (Swagger) and a ReDoc view at `/redoc` when the server is running. The Swagger UI includes per-route summaries and brief usage notes for the key endpoints (`/sessions`, `/leaderboard`, `/ingest`).

To run the app locally and open Swagger:

```powershell
uvicorn server.main:app --reload
# then open https://127.0.0.1:8000/docs in your browser
```

Internal observability (lightweight)
------------------------------------

- `GET /internal/metrics`: returns in-memory counters, timer summaries, and recent orchestration traces.
- `POST /internal/metrics/reset`: clears in-memory metrics/traces.
- If `ADMIN_TOKEN` is set, pass `admin_token` query param for these internal routes.

This project intentionally uses lightweight in-process observability for current workload size (no external agent required yet).

Ingestion formats
-----------------

- `POST /ingest/book`: ingest pre-extracted plain text.
- `POST /ingest/file`: upload and ingest `.txt`, `.pdf`, or `.epub` directly.
- `POST /ingest/books/{book_id}/topics` defaults to `mode=rag` (pass `mode=rule` for deterministic fallback).
- Topic creation with `mode=rag` uses orchestrator topic proposals; with `ORCHESTRATOR_DRAFT_MODE=llm`, topic titles are LLM-generated when available.
