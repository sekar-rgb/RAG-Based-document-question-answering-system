from retriever import load_index_and_metadata, retrieve
from generator import generate_answer

if __name__ == '__main__':
    print("[1] Loading index and metadata...")
    try:
        index, metadata = load_index_and_metadata()
        print(f"[OK] Loaded {len(metadata)} metadata entries")
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        exit(1)

    query = "What is the main idea?"
    print(f"[2] Retrieving context for query: {query}")
    try:
        results = retrieve(query, index, metadata)
        print(f"[OK] Retrieved {len(results)} chunks")
        for r in results:
            print(f"  - Source: {r.get('source')}, Score: {r.get('score'):.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to retrieve: {e}")
        exit(1)

    print(f"[3] Generating answer...")
    try:
        answer = generate_answer(query, results)
        print(f"[OK] Answer:\n{answer}")
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
