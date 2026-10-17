import argparse
import sys
from typing import Optional

from config import INDEX_DIR, INDEX_PATH, METADATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

from ingest import ingest_folder
from retriever import load_index_and_metadata, retrieve
from generator import generate_answer


def cmd_ingest(args: argparse.Namespace) -> None:
    folder = args.docs
    chunk_size = args.chunk_size or CHUNK_SIZE
    overlap = args.overlap or CHUNK_OVERLAP
    print(f"Ingesting documents from {folder} (chunk_size={chunk_size}, overlap={overlap})")
    ingest_folder(folder, chunk_size, overlap)
    print("Ingestion complete. Index and metadata saved.")


def cmd_query(args: argparse.Namespace) -> None:
    query = args.query
    try:
        index, metadata = load_index_and_metadata()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)

    results = retrieve(query, index, metadata)
    print(f"Retrieved {len(results)} context blocks. Sending to LLM...")
    answer = generate_answer(query, results)
    print('\n--- Answer ---\n')
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(prog='rag-app')
    sub = parser.add_subparsers(dest='cmd')

    p_ingest = sub.add_parser('ingest')
    p_ingest.add_argument('--docs', required=True, help='Path to folder with .txt/.pdf files')
    p_ingest.add_argument('--chunk_size', type=int, default=None)
    p_ingest.add_argument('--overlap', type=int, default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser('query')
    p_query.add_argument('query', help='Question to ask the RAG system')
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    if not hasattr(args, 'func') or args.cmd is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
