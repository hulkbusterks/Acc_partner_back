from server.services.ingest_manager import IngestManager
from server.services.faiss_store import store as faiss_store


def test_rag_topics_created_when_vector_store_cold():
    ing = IngestManager()
    book = ing.create_book(
        title="Cold Store Book",
        authors="Tester",
        raw_text="Chapter one introduces reliability. Chapter two discusses retries and observability.",
    )

    # Simulate app restart/cold memory vector store.
    faiss_store.items.clear()
    faiss_store.id_list.clear()

    topics = ing.create_topics_from_book(book.id, mode="rag")
    assert len(topics) >= 1


def test_rule_topics_follow_section_titles():
    ing = IngestManager()
    book = ing.create_book(
        title="Rule Headings Book",
        authors="Tester",
        raw_text=(
            "CHAPTER 1\n"
            "Intro to reliability and retries.\n\n"
            "CHAPTER 2\n"
            "Practice observability and incident response."
        ),
    )

    topics = ing.create_topics_from_book(book.id, mode="rule")
    titles = [t.title for t in topics]

    assert any("chapter 1" in (title or "").lower() for title in titles)
    assert any("chapter 2" in (title or "").lower() for title in titles)