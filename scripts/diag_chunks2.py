"""Diagnose full chunk coverage for largest books."""
from server.db import create_db_and_tables, get_session
from server.models import Topic, Book
from sqlmodel import select

create_db_and_tables()
with get_session() as db:
    # Find books with actual content
    books = db.exec(select(Book).order_by(Book.title)).all()
    for b in books:
        raw_len = len(b.raw_text or "")
        if raw_len < 50:
            continue
        from server.services.ingest_manager import ingest_manager
        chunks = ingest_manager.chunk_text(b.raw_text or "")
        sections = (b.meta or {}).get("sections", [])
        print(f"\nBook: \"{b.title[:50]}\" id={b.id[:8]} raw={raw_len} chars, {len(chunks)} chunks, {len(sections)} sections")
        for s in sections:
            print(f"  Section: {s['title'][:40]:<40} chunks {s['start']}-{s['end']}")
        
        # Find topics for this book
        topics = db.exec(select(Topic).where(Topic.book_id == b.id)).all()
        for t in topics:
            sc = t.source_chunks or {}
            ids = sc.get("ids", [])
            indexes = sc.get("indexes", [])
            total_assigned = len(ids) + len(indexes)
            print(f"  -> Topic \"{t.title[:30]}\" has {total_assigned}/{len(chunks)} chunks assigned")
            if ids:
                # Extract chunk indexes from IDs
                idx_nums = sorted(int(x.split("::")[-1]) for x in ids if "::" in x)
                print(f"     Chunk indexes: {idx_nums}")
