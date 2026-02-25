from server.services.ingest_manager import IngestManager
from server.services.session_manager import manager as session_manager
from server.db import get_session
from server.models import Book, Session, CachedQuestion, PromptEvent
from sqlmodel import select

# create a small book
ing = IngestManager()
text = """
This is a short sample text about programming. Python is a popular language. Unit tests help ensure code correctness.
The quick brown fox jumps over the lazy dog. Sample content for FAISS testing.
"""
book = ing.create_book(title="Sample Book", authors="Tester", raw_text=text)
print('Inserted book id:', book.id)

# DB schema should be managed by Alembic migrations. If you haven't run
# migrations yet, run the alembic upgrade (see project README).

sess = session_manager.create_session(user_id='user-1', topic_id=book.id)
print('Created session id:', sess.id)

# generate prompts (mock orchestrator)
res = session_manager.generate_prompts(sess.id, n=3)
print('Generate prompts returned:', res)

# show persisted CachedQuestion and PromptEvent rows
db = get_session()
qs = db.exec(select(CachedQuestion.id, CachedQuestion.question_json, CachedQuestion.variants)).all()
pe = db.exec(select(PromptEvent.id, PromptEvent.prompt_text, PromptEvent.prompt_question_id, PromptEvent.created_at)).all()
print('CachedQuestion rows:')
for q in qs:
    print(q)
print('PromptEvent rows:')
for p in pe:
    print(p)
