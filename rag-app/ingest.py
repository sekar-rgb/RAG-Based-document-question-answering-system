import os
import json
from typing import List, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from PyPDF2 import PdfReader
from docx import Document

from config import INDEX_DIR, INDEX_PATH, METADATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def ensure_index_dir() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)


def load_text_from_file(path: str) -> str:
    if path.lower().endswith('.pdf'):
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    elif path.lower().endswith('.docx'):
        doc = Document(path)
        texts = [para.text for para in doc.paragraphs]
        return "\n".join(texts)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def chunk_text(text: str, tokenizer: AutoTokenizer, chunk_size: int, overlap: int) -> List[str]:
    # Try tokenization-based chunking; fall back to simple whitespace chunking on failure
    chunks: List[str] = []
    try:
        token_ids = tokenizer.encode(text)
        step = max(1, chunk_size - overlap)
        for i in range(0, max(1, len(token_ids)), step):
            chunk_ids = token_ids[i:i + chunk_size]
            chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if chunk.strip():
                chunks.append(chunk.strip())
        if chunks:
            return chunks
    except Exception:
        # fall through to whitespace fallback
        pass

    # Fallback: approximate tokens with words when tokenizer fails or returns nothing
    words = text.split()
    if not words:
        return []
    step_w = max(1, chunk_size - overlap)
    for i in range(0, len(words), step_w):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_folder(folder_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Tuple[Optional[faiss.IndexFlatL2], List[dict]]:
    ensure_index_dir()
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_chunks: List[str] = []
    metadatas: List[dict] = []
    files_processed = 0
    processed_files = []
    failed_files = []
    all_files_seen = []

    for root, _, files in os.walk(folder_path):
        for fname in files:
            path = os.path.join(root, fname)
            all_files_seen.append(path)
            if not (fname.lower().endswith('.txt') or fname.lower().endswith('.pdf') or fname.lower().endswith('.docx')):
                continue
            try:
                text = load_text_from_file(path)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                failed_files.append(path)
                continue
            text_len = len(text) if text is not None else 0
            chunks = chunk_text(text or "", tokenizer, chunk_size, overlap)
            print(f"Processing file: {path} (chars={text_len}) -> chunks={len(chunks)}")
            files_processed += 1
            processed_files.append(path)
            for c in chunks:
                metadatas.append({"text": c, "source": fname})
                all_chunks.append(c)

    print(f"\n--- Ingestion Report ---")
    print(f"Total files found in directory: {len(all_files_seen)}")
    print(f"Files processed successfully: {len(processed_files)}")
    for f in processed_files:
        print(f"  - [SUCCESS] {f}")
    if failed_files:
        print(f"Files failed to load: {len(failed_files)}")
        for f in failed_files:
            print(f"  - [FAILED] {f}")
    print(f"------------------------\n")

    if not all_chunks:
        print(f"Warning: No text chunks found in {folder_path}. Files inspected: {processed_files}")
        return None, []

    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    return index, metadatas


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', required=True, help='Path to folder with .txt/.pdf files')
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--overlap', type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    ingest_folder(args.docs, args.chunk_size, args.overlap)
