"""Diagnose chunk coverage for topics."""
from server.db import create_db_and_tables, get_session
from server.models import Topic, Book
from sqlmodel import select

create_db_and_tables()
with get_session() as db:
    topics = db.exec(select(Topic).limit(10)).all()
    for t in topics:
        sc = t.source_chunks or {}
        ids = sc.get("ids", [])
        indexes = sc.get("indexes", [])
        print(f"Topic: {t.title[:40]:<40} book={t.book_id[:8]}  ids={len(ids)} indexes={len(indexes)}  ids_sample={ids[:5]}  idx_sample={indexes[:5]}")

    if topics:
        book = db.get(Book, topics[0].book_id)
        if book and book.raw_text:
            from server.services.ingest_manager import ingest_manager
            chunks = ingest_manager.chunk_text(book.raw_text)
            print(f"\nBook \"{book.title[:40]}\" has {len(chunks)} total chunks, raw_text={len(book.raw_text)} chars")
            sections = (book.meta or {}).get("sections", [])
            print(f"Sections: {len(sections)}")
            for s in sections:
                print(f"  {s['title'][:40]:<40} chunks {s['start']}-{s['end']}")
