# Accountability Backend — API Reference

> **Base URL:** `http://localhost:8000` (dev) — configurable via `uvicorn`  
> **OpenAPI UI:** `GET /docs` (Swagger) or `GET /redoc` (ReDoc)  
> **Version:** 0.2.0

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Content Ingestion](#2-content-ingestion)
3. [Study Sessions](#3-study-sessions)
4. [Leaderboard](#4-leaderboard)
5. [Internal / Observability](#5-internal--observability)
6. [Health Check](#6-health-check)
7. [Common Patterns](#7-common-patterns)
8. [Error Reference](#8-error-reference)
9. [Data Models](#9-data-models)
10. [Session Lifecycle Flowchart](#10-session-lifecycle-flowchart)

---

## 1. Authentication

All endpoints (except `/auth/*` and `/health`) require a Bearer token in the `Authorization` header.

### `POST /auth/register`

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secureP@ss1",
  "display_name": "Jane Doe"       // optional
}
```

**Validation:**
- `email` — must match `^[^@\s]+@[^@\s]+\.[^@\s]+$`
- `password` — minimum 8 characters

**Response `200`:**
```json
{
  "user_id": "b0f6a7b8-b575-4e5e-ae1c-03c69894b435"
}
```

**Errors:** `400` Invalid email / Password too short / Email already registered

---

### `POST /auth/login`

Obtain a JWT Bearer token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secureP@ss1"
}
```

**Response `200`:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user_id": "b0f6a7b8-b575-4e5e-ae1c-03c69894b435"
}
```

**Token Details:**
- Algorithm: HS256
- Expiry: 24 hours
- Payload: `{ "user_id": "...", "email": "..." }`

**Errors:** `401` Invalid credentials

---

### `GET /auth/users`

List all registered users. Does not expose `password_hash` or `preferences`.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type | Default | Description    |
|--------|------|---------|----------------|
| `limit`| int  | `50`    | Max results    |

**Response `200`:**
```json
[
  {
    "id": "b0f6a7b8-...",
    "email": "user@example.com",
    "display_name": "Jane Doe",
    "created_at": "2026-02-24 10:00:00+00:00"
  }
]
```

---

### `GET /auth/users/me`

Get the authenticated user's own profile.

**Headers:** `Authorization: Bearer <token>`

**Response `200`:**
```json
{
  "id": "b0f6a7b8-...",
  "email": "user@example.com",
  "display_name": "Jane Doe",
  "created_at": "2026-02-24 10:00:00+00:00"
}
```

---

### `GET /auth/users/{user_id}`

Get a specific user's profile by their ID.

**Headers:** `Authorization: Bearer <token>`

**Response `200`:**
```json
{
  "id": "b0f6a7b8-...",
  "email": "user@example.com",
  "display_name": "Jane Doe",
  "created_at": "2026-02-24 10:00:00+00:00"
}
```

**Errors:** `404` User not found

---

## 2. Content Ingestion

### `POST /ingest/book`

Ingest a book from raw text. The backend chunks the text and indexes it into the FAISS vector store.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "title": "Introduction to Algorithms",
  "authors": "Cormen et al.",         // optional
  "raw_text": "Full text content..."  // optional
}
```

**Response `200`:**
```json
{
  "book_id": "a1b2c3d4-..."
}
```

---

### `POST /ingest/file`

Upload a file (PDF, EPUB, TXT) and ingest its contents.

**Headers:** `Authorization: Bearer <token>`

**Content-Type:** `multipart/form-data`

| Field    | Type         | Required | Description                      |
|----------|--------------|----------|----------------------------------|
| `title`  | string       | Yes      | Book title                       |
| `file`   | UploadFile   | Yes      | The file to ingest               |
| `authors`| string       | No       | Author name(s)                   |

**Limits:** Max 50 MB

**Response `200`:**
```json
{
  "book_id": "a1b2c3d4-...",
  "chars": 154230
}
```

**Errors:** `400` Empty file / Unsupported file type / No readable text · `413` File too large

---

### `POST /ingest/books/{book_id}/topics`

Auto-generate study topics from a book's content using RAG-based topic proposal or rule-based section splitting.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type   | Default | Description                                      |
|--------|--------|---------|--------------------------------------------------|
| `mode` | string | `"rag"` | `"rag"` (LLM-assisted) or `"rule"` (section-based) |

**Response `200`:**
```json
{
  "created": 5,
  "topics": [
    { "id": "t1-uuid", "title": "Binary Search Trees" },
    { "id": "t2-uuid", "title": "Graph Traversal" }
  ]
}
```

---

### `GET /ingest/books`

List all ingested books.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type | Default | Description    |
|--------|------|---------|----------------|
| `limit`| int  | `50`    | Max results    |

**Response `200`:**
```json
[
  {
    "id": "a1b2c3d4-...",
    "title": "Introduction to Algorithms",
    "authors": "Cormen et al."
  }
]
```

---

### `GET /ingest/books/{book_id}/topics`

List all topics generated for a specific book.

**Headers:** `Authorization: Bearer <token>`

**Response `200`:**
```json
[
  {
    "id": "t1-uuid",
    "book_id": "a1b2c3d4-...",
    "title": "Binary Search Trees",
    "start_page": null,
    "end_page": null
  }
]
```

---

## 3. Study Sessions

### Session Lifecycle

```
create → start → generate_questions → [next_question → submit]* → end
```

All session endpoints enforce **ownership** — only the user who created a session can interact with it.

---

### `POST /sessions/`

Create a new study session for the authenticated user.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "topic_id": "t1-uuid",
  "requested_minutes": 30,    // optional, default 30
  "tone": "neutral"           // optional: "neutral" | "mean"
}
```

**Response `200`:**
```json
{
  "session_id": "s1-uuid"
}
```

---

### `POST /sessions/{session_id}/start`

Start the session timer. Must be called before generating questions or submitting answers.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param              | Type | Default | Description          |
|--------------------|------|---------|----------------------|
| `duration_minutes` | int  | `30`    | Timer duration in minutes |

**Response `200`:**
```json
{
  "session_id": "s1-uuid",
  "started_at": "2026-02-24T10:00:00Z"
}
```

**Errors:** `404` Session not found · `403` Not allowed

---

### `POST /sessions/{session_id}/generate_questions`

Generate MCQ questions from the session's topic context. Uses the RAG orchestrator to retrieve relevant chunks and build validated MCQs.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param | Type | Default | Description                   |
|-------|------|---------|-------------------------------|
| `n`   | int  | `5`     | Number of MCQs to generate    |

**Rate Limit:** Default 30 calls/min per user (env `ORCHESTRATOR_RATE_LIMIT_PER_MIN`)

**Response `200`:**
```json
{
  "generated": 5,
  "questions": [
    {
      "question_json": {
        "question": "What data structure does a priority queue typically use?",
        "choices": ["Binary heap", "Linked list", "Hash table", "Stack"],
        "correct_index": 0,
        "explanation": "A binary heap provides O(log n) insert and extract-min."
      },
      "prompt_id": "pe-uuid",
      "source_chunks": ["chunk-id-1", "chunk-id-2"],
      "verified": true,
      "deterministic_supported": true,
      "graph_meta": { ... }
    }
  ]
}
```

**Question JSON shape:**

| Field           | Type     | Description                                    |
|-----------------|----------|------------------------------------------------|
| `question`      | string   | The question text                              |
| `choices`       | string[] | Exactly 4 answer choices                       |
| `correct_index` | int      | 0-based index of the correct choice            |
| `explanation`   | string   | Why the correct answer is correct              |

**Errors:** `429` Rate limit · `403` Not owner · `ValueError` Session not started / expired

---

### `GET /sessions/{session_id}/next_question`

Get the next unanswered MCQ for this session (randomly selected from the unanswered pool). Returns the full MCQ with question text, 4 choices, and a prompt_id for submitting. The correct answer is **not** included — it is revealed after submission.

**Headers:** `Authorization: Bearer <token>`

**Response `200` (prompt available):**
```json
{
  "next": {
    "prompt_id": "pe-uuid",
    "prompt_text": "What data structure does a priority queue typically use?",
    "question": "What data structure does a priority queue typically use?",
    "choices": ["Binary heap", "Linked list", "Hash table", "Stack"],
    "remaining": 4
  }
}
```

| Field        | Type     | Description                                 |
|--------------|----------|---------------------------------------------|
| `prompt_id`  | string   | Use this to submit the answer               |
| `prompt_text`| string   | The question text (legacy field)             |
| `question`   | string   | The question text                            |
| `choices`    | string[] | The 4 answer options (0-indexed)             |
| `remaining`  | int      | Number of unanswered questions left          |

> **Note:** `correct_index` is intentionally omitted — the answer is revealed only after calling `POST /submit`.

**Response `200` (no more prompts or session expired):**
```json
{
  "next": null
}
```

---

### `POST /sessions/{session_id}/submit`

Submit an answer (or reject a question) for the given prompt. The response reveals whether the answer was correct, the correct answer, an explanation, and per-option reasoning.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "prompt_id": "pe-uuid",
  "answer": "0",           // 0-based index of chosen option, OR text match
  "reject": false          // optional, default false — set true to skip
}
```

**Answer Matching:**
- **By index (preferred):** Pass `"0"`, `"1"`, `"2"`, or `"3"` (0-based, matching `correct_index`)
- **By text:** Pass the exact text of a choice (case-insensitive)

**Response `200` (normal answer):**
```json
{
  "correct": true,
  "session_score": 3,
  "failures": 1,
  "correct_index": 0,
  "correct_answer": "Binary heap",
  "explanation": "A binary heap provides O(log n) insert and extract-min.",
  "options": [
    { "index": 0, "text": "Binary heap", "correct": true, "reason": "A binary heap provides O(log n) insert and extract-min." },
    { "index": 1, "text": "Linked list", "correct": false, "reason": "This option is incorrect." },
    { "index": 2, "text": "Hash table", "correct": false, "reason": "This option is incorrect." },
    { "index": 3, "text": "Stack", "correct": false, "reason": "This option is incorrect." }
  ]
}
```

| Field           | Type     | Description                                    |
|-----------------|----------|------------------------------------------------|
| `correct`          | bool     | Whether the submitted answer was correct       |
| `session_score`    | int      | Running total of correct answers               |
| `failures`         | int      | Running total of wrong answers                 |
| `correct_index`    | int      | 0-based index of the correct choice            |
| `correct_answer`   | string   | Text of the correct choice                     |
| `explanation`      | string   | Why the correct answer is correct              |
| `options`          | array    | Per-option breakdown (see below)               |
| `remaining`        | int      | Number of unanswered questions left            |
| `session_complete` | bool     | `true` when all questions have been answered   |

**Option object:**

| Field    | Type   | Description                         |
|----------|--------|-------------------------------------|
| `index`  | int    | 0-based position in choices array   |
| `text`   | string | The choice text                     |
| `correct`| bool   | Whether this is the right answer    |
| `reason` | string | Why this option is right or wrong   |

**Response `200` (with "mean" tone after 3+ failures+rejects):**
```json
{
  "correct": false,
  "session_score": 0,
  "failures": 3,
  "correct_index": 0,
  "correct_answer": "Binary heap",
  "explanation": "...",
  "options": [ ... ],
  "mean_comment": "Seriously? You're doing terribly. Try focusing."
}
```

**Response `200` (rejected):**
```json
{
  "rejected": true,
  "session_rejects": 2
}
```

**Errors:**
- `ValueError` — prompt not found / prompt does not belong to session / prompt already answered / session not started / session expired
- `403` Not owner

---

### `POST /sessions/{session_id}/end`

End the session, finalize the score, and upsert the user's leaderboard aggregate.

**Headers:** `Authorization: Bearer <token>`

**Response `200`:**
```json
{
  "session_id": "s1-uuid",
  "ended_at": "2026-02-24T10:30:00Z",
  "score": 4,
  "aggregate": {
    "user_id": "u-uuid",
    "best_score": 4,
    "total_score": 12,
    "sessions": 3
  }
}
```

> `aggregate` is `null` if leaderboard upsert fails.

**Errors:** `ValueError` — session not found / session already ended · `403` Not owner

---

## 4. Leaderboard

### `POST /leaderboard/entries`

Create a raw leaderboard entry for the authenticated user.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type  | Required | Description |
|--------|-------|----------|-------------|
| `score`| float | Yes      | Score value |

**Response `200`:** Returns `LeaderboardEntry` object (see [Data Models](#9-data-models)).

---

### `GET /leaderboard/entries`

List raw leaderboard entries, sorted by score descending.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type | Default | Description    |
|--------|------|---------|----------------|
| `limit`| int  | `50`    | Max results    |

**Response `200`:** Array of `LeaderboardEntry` objects.

---

### `GET /leaderboard/entries/me`

Get the current user's score history (audit trail). Returns all `LeaderboardEntry` rows for the authenticated user, newest first.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type | Default | Description    |
|--------|------|---------|----------------|
| `limit`| int  | `50`    | Max results    |

**Response `200`:** Array of `LeaderboardEntry` objects.

---

### `GET /leaderboard/aggregate/top`

Return top raw entries by score (alias for entries with default limit 10).

**Headers:** `Authorization: Bearer <token>`

**Query Params:** `limit` (int, default 10)

**Response `200`:** Array of `LeaderboardEntry` objects.

---

### `POST /leaderboard/submit`

Submit a session result. Upserts the per-user aggregate (best score, total score, session count). Uses the authenticated user's ID automatically.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param  | Type  | Required | Description  |
|--------|-------|----------|--------------|
| `score`| float | Yes      | Session score|

**Response `200`:** Returns `LeaderboardAggregate` object.

---

### `GET /leaderboard/aggregates`

List per-user aggregate rankings.

**Headers:** `Authorization: Bearer <token>`

**Query Params:**

| Param     | Type   | Default  | Description                     |
|-----------|--------|----------|---------------------------------|
| `limit`   | int    | `50`     | Max results                     |
| `order_by`| string | `"best"` | `"best"` or `"total"`          |

**Response `200`:** Array of `LeaderboardAggregate` objects.

---

### `GET /leaderboard/aggregate/{user_id}`

Fetch a specific user's aggregate stats (cached with short in-process TTL).

**Headers:** `Authorization: Bearer <token>`

**Response `200`:** `LeaderboardAggregate` object.

**Errors:** `404` Not found

---

### `POST /leaderboard/reset`

**Admin-only.** Clear all leaderboard data and cache.

**Headers:** `X-Admin-Token: <admin_token>` (must match env `ADMIN_TOKEN`)

**Response `200`:**
```json
{ "status": "ok" }
```

**Errors:** `401` Admin token not configured / Unauthorized

---

## 5. Internal / Observability

### `GET /internal/metrics`

Returns in-memory counters, timer summaries, and recent orchestration traces.

**Headers (optional):** `X-Admin-Token: <token>` (required if `ADMIN_TOKEN` env is set)

**Response `200`:**
```json
{
  "counters": {
    "orchestrator_generate_attempt_total": 42,
    "orchestrator_generate_success_total": 40,
    "orchestrator_generate_error_total": 2,
    "orchestrator_generate_rate_limited_total": 0
  },
  "timers": {
    "orchestrator_generate_latency_ms": {
      "count": 40,
      "mean": 1250.5,
      "max": 3200.1
    }
  },
  "traces": [ ... ]
}
```

---

### `POST /internal/metrics/reset`

Clear all in-memory counters and traces.

**Headers (optional):** `X-Admin-Token: <token>`

**Response `200`:**
```json
{ "status": "ok" }
```

---

## 6. Health Check

### `GET /health`

Returns system health status. No authentication required.

**Response `200`:**
```json
{
  "status": "ok",
  "db": "ok",
  "faiss_vectors": 1024,
  "faiss_items": 256
}
```

| Field          | Description                                      |
|----------------|--------------------------------------------------|
| `status`       | `"ok"` or `"degraded"`                           |
| `db`           | `"ok"` or `"error: <details>"`                   |
| `faiss_vectors`| Number of FAISS index vectors loaded             |
| `faiss_items`  | Number of indexed text chunks                    |

---

## 7. Common Patterns

### Authentication

All protected endpoints use the same pattern:

```
Authorization: Bearer <jwt_token>
```

The token is obtained from `POST /auth/login`. It expires after **24 hours**.

### Error Responses

All errors follow this shape:

```json
{
  "detail": "Human-readable error message"
}
```

### Ownership

Session endpoints enforce ownership. You can only interact with sessions you created. Attempting to access another user's session returns `403 Not Allowed`.

### Rate Limiting

Question generation is rate-limited to prevent LLM abuse. Default: 30 requests/minute per user. Returns `429` when exceeded.

---

## 8. Error Reference

| Code | Meaning                    | Typical Cause                                         |
|------|----------------------------|-------------------------------------------------------|
| 400  | Bad Request                | Validation failure (email format, empty file, etc.)   |
| 401  | Unauthorized               | Missing/invalid token, wrong admin token              |
| 403  | Forbidden                  | Trying to access another user's session               |
| 404  | Not Found                  | Session, aggregate, or resource doesn't exist         |
| 413  | Payload Too Large          | Uploaded file exceeds 50 MB limit                     |
| 429  | Too Many Requests          | Rate limit exceeded for question generation           |
| 500  | Internal Server Error      | Unhandled exception (logged server-side)              |

---

## 9. Data Models

### User

| Field         | Type             | Description                          |
|---------------|------------------|--------------------------------------|
| `id`          | string (UUID)    | Primary key                          |
| `email`       | string           | Unique email address                 |
| `display_name`| string \| null   | Optional display name                |
| `created_at`  | datetime (UTC)   | Registration timestamp               |

### Book

| Field    | Type             | Description                          |
|----------|------------------|--------------------------------------|
| `id`     | string (UUID)    | Primary key                          |
| `title`  | string           | Book title                           |
| `authors`| string \| null   | Author(s)                            |
| `meta`   | object           | Additional metadata (JSON)           |

### Topic

| Field          | Type             | Description                          |
|----------------|------------------|--------------------------------------|
| `id`           | string (UUID)    | Primary key                          |
| `book_id`      | string (UUID)    | FK → Book                           |
| `title`        | string           | Topic title                          |
| `start_page`   | int \| null      | Start page reference                 |
| `end_page`     | int \| null      | End page reference                   |
| `source_chunks`| object           | `{ "indexes": [0, 1, 2] }` chunk refs |

### Session

| Field          | Type             | Description                          |
|----------------|------------------|--------------------------------------|
| `id`           | string (UUID)    | Primary key                          |
| `user_id`      | string (UUID)    | FK → User (owner)                   |
| `topic_id`     | string (UUID)    | FK → Topic                          |
| `started_at`   | datetime \| null | When `start` was called              |
| `ended_at`     | datetime \| null | When `end` was called                |
| `duration_sec` | int \| null      | Timer duration in seconds            |
| `failure_count`| int              | Wrong answers count                  |
| `reject_count` | int              | Rejected questions count             |
| `score`        | int              | Correct answers count                |
| `tone`         | string           | `"neutral"` or `"mean"`              |

### LeaderboardEntry

Audit-trail row: one per finished session (created automatically by `POST /sessions/{id}/end`).

| Field          | Type             | Description                          |
|----------------|------------------|--------------------------------------|
| `id`           | string (UUID)    | Primary key                          |
| `user_id`      | string (UUID)    | FK → User                           |
| `session_id`   | string (UUID)    | FK → Session (nullable)             |
| `topic_id`     | string (UUID)    | FK → Topic (nullable)               |
| `book_id`      | string (UUID)    | FK → Book (nullable)                |
| `score`        | float            | Score value                          |
| `session_count`| int              | Default 1                            |
| `created_at`   | datetime (UTC)   | Created timestamp                    |
| `updated_at`   | datetime (UTC)   | Last updated                         |

### LeaderboardAggregate

| Field        | Type             | Description                          |
|--------------|------------------|--------------------------------------|
| `user_id`    | string (UUID)    | PK + FK → User                      |
| `best_score` | float            | Highest single-session score         |
| `total_score`| float            | Sum of all session scores            |
| `sessions`   | int              | Number of sessions completed         |
| `updated_at` | datetime (UTC)   | Last updated                         |

---

## 10. Session Lifecycle Flowchart

```
┌─────────────┐
│  Register   │  POST /auth/register
│  & Login    │  POST /auth/login  →  get JWT token
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Ingest Book │  POST /ingest/book  or  POST /ingest/file
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Create      │  POST /ingest/books/{id}/topics
│ Topics      │  → returns topic IDs to use in sessions
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Create      │  POST /sessions/
│ Session     │  body: { topic_id, requested_minutes, tone }
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Start       │  POST /sessions/{id}/start?duration_minutes=30
│ Session     │  → starts the countdown timer
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Generate    │  POST /sessions/{id}/generate_questions?n=5
│ Questions   │  → returns normalized MCQs with prompt_ids
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────┐
│              Quiz Loop                    │
│                                          │
│  GET  /sessions/{id}/next_question       │
│  → { question, choices, prompt_id,       │
│      remaining }                         │
│                                          │
│  POST /sessions/{id}/submit              │
│  body: { prompt_id, answer: "0" }        │
│  → { correct, correct_answer, options,   │
│      remaining, session_complete }       │
│                                          │
│  (repeat until session_complete == true) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌─────────────┐
│ End Session │  POST /sessions/{id}/end
│             │  → { score, aggregate }
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Leaderboard │  GET /leaderboard/aggregates
│ Check       │  GET /leaderboard/aggregate/{user_id}
└─────────────┘
```

---

## Environment Variables

| Variable                         | Default          | Description                                    |
|----------------------------------|------------------|------------------------------------------------|
| `JWT_SECRET`                     | (required)       | Secret key for JWT signing                     |
| `DATABASE_URL`                   | `sqlite:///app.db`| SQLAlchemy connection string                  |
| `CORS_ORIGINS`                   | `*`              | Comma-separated allowed origins                |
| `LLM_PROVIDER`                   | `mock`           | `mock`, `azure`, `openai`                      |
| `ORCHESTRATOR`                   | `mock`           | `mock` or `langgraph`                          |
| `ORCHESTRATOR_RATE_LIMIT_PER_MIN`| `30`             | Rate limit for question generation             |
| `ORCHESTRATOR_DRAFT_MODE`        | (unset)          | `llm` to use LLM for MCQ drafting              |
| `ORCHESTRATOR_VERIFY`            | `false`          | `true` to enable context quality verification  |
| `EMBEDDING_PROVIDER`             | `auto`           | `hf`, `adapter`, or `auto`                     |
| `EMBEDDING_MODEL`                | `sentence-transformers/all-MiniLM-L6-v2` | HF model name |
| `EMBEDDING_WARMUP`               | `false`          | `true` to warmup embeddings on startup         |
| `ADMIN_TOKEN`                    | (unset)          | Token for admin endpoints (reset, metrics)     |
| `FAISS_PERSIST_DIR`              | (unset)          | Directory to persist FAISS index to disk        |
