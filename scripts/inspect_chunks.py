import sys
from server.services.ingest_manager import IngestManager
from server.services.faiss_store import store as faiss_store

def main():
    text = None
    query = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print('Failed to read file', e)
            return
    else:
        text = """
This is a short sample text about programming. Python is a popular language. Unit tests help ensure code correctness.
The quick brown fox jumps over the lazy dog. Sample content for FAISS testing.
"""

    if len(sys.argv) > 2:
        query = sys.argv[2]
    else:
        query = "python tests"

    ing = IngestManager()
    print('=== Tokenization / chunk preview ===')
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        toks = enc.encode(text)
        print('Total tokens:', len(toks))
    except Exception:
        print('tiktoken not available — falling back to character-based estimates')

    chunks = ing.chunk_text(text)
    print('Produced', len(chunks), 'chunks:')
    for i, c in enumerate(chunks):
        print(f'--- chunk {i} (len={len(c)} chars) ---')
        print(c[:400].replace('\n', ' '))

    # upsert into store with temporary ids
    ids = [f'temp::chunk::{i}' for i in range(len(chunks))]
    faiss_store.upsert(ids, chunks)

    print('\n=== Retrieval preview for query:', query, '===')
    res = faiss_store.query(query, k=5)
    for rid, score in res:
        print('-', rid, score)

if __name__ == '__main__':
    main()
