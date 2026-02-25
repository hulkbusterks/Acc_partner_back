import time
import sys
from server.services.ingest_manager import IngestManager
from server.services.faiss_store import store as faiss_store


def generate_long_text(n_paragraphs: int = 200):
    para = (
        "This is a sample paragraph about programming and testing. "
        "Python is a language used for prototyping and tests. "
        "Unit tests and integration tests help maintain code quality. "
    )
    return "\n\n".join([para for _ in range(n_paragraphs)])


def bench(text: str, chunk_tokens_vals, chunk_size_vals):
    results = []
    for ct in chunk_tokens_vals:
        for cs in chunk_size_vals:
            ing = IngestManager(chunk_tokens=ct, chunk_size=cs)
            start = time.perf_counter()
            chunks = ing.chunk_text(text)
            t_chunk = time.perf_counter() - start
            # upsert into store and measure query
            ids = [f"bench::{i}" for i in range(len(chunks))]
            faiss_store.items.clear()
            faiss_store.id_list.clear()
            up_start = time.perf_counter()
            faiss_store.upsert(ids, chunks)
            up_time = time.perf_counter() - up_start
            q_start = time.perf_counter()
            res = faiss_store.query("python tests", k=3)
            q_time = time.perf_counter() - q_start
            results.append((ct, cs, len(chunks), t_chunk, up_time, q_time))
            print(f"ct={ct} cs={cs} chunks={len(chunks)} chunk_time={t_chunk:.4f}s up={up_time:.4f}s q={q_time:.4f}s")
    return results


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = generate_long_text(400)

    chunk_tokens_vals = [100, 200, 400]
    chunk_size_vals = [800, 1200, 2000]
    res = bench(text, chunk_tokens_vals, chunk_size_vals)
    print('\nSummary:')
    for r in res:
        print(r)


if __name__ == '__main__':
    main()
