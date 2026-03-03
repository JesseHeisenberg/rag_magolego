import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from api import openrouter_api


DEFAULT_PDF_PATH = "document.pdf"
DEFAULT_INDEX_DIR = ".rag_index"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"  # любой доступный вам в OpenRouter
DEFAULT_TOP_K = 5

# OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class Chunk:
    text: str
    meta: dict


def read_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    """Возвращает список (page_number, text)"""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = re.sub(r"\s+\n", "\n", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = txt.strip()
        if txt:
            pages.append((i, txt))
    return pages


def chunk_text(
    pages: List[Tuple[int, str]],
    chunk_size: int = 900,
    chunk_overlap: int = 150
) -> List[Chunk]:
    """
    Простой чанкер по символам с overlap.
    chunk_size и overlap в символах (быстро и надёжно для PDF).
    """
    chunks: List[Chunk] = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(Chunk(text=chunk, meta={"page": page_num}))
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
    return chunks


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(
    pdf_path: str,
    index_dir: str,
    embed_model_name: str = DEFAULT_EMBED_MODEL
):
    ensure_dir(index_dir)

    pages = read_pdf_text(pdf_path)
    if not pages:
        raise RuntimeError("Не удалось извлечь текст из PDF (возможно, это скан).")

    chunks = chunk_text(pages)

    embedder = SentenceTransformer(embed_model_name)
    texts = [c.text for c in chunks]

    # embeddings: (N, D) float32
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity при normalize_embeddings=True
    index.add(embs)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    meta = {
        "pdf_path": pdf_path,
        "embed_model": embed_model_name,
        "chunks": [{"text": c.text, "meta": c.meta} for c in chunks],
        "dim": int(dim),
    }
    save_json(os.path.join(index_dir, "meta.json"), meta)


def load_index(index_dir: str):
    idx_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Индекс не найден. Сначала запустите команду build.")

    index = faiss.read_index(idx_path)
    meta = load_json(meta_path)
    return index, meta


def retrieve(
    query: str,
    index,
    meta: dict,
    embed_model_name: str,
    top_k: int = DEFAULT_TOP_K
) -> List[dict]:
    embedder = SentenceTransformer(embed_model_name)
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    scores, ids = index.search(q, top_k)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    results = []
    for i, s in zip(ids, scores):
        if i == -1:
            continue
        chunk = meta["chunks"][i]
        results.append({
            "score": float(s),
            "text": chunk["text"],
            "meta": chunk["meta"],
        })
    return results


def openrouter_chat(
    messages: list,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
):
    headers = {
        "Authorization": f"Bearer {openrouter_api}",
        "Content-Type": "application/json",
        # По желанию (не обязательно):
        # "HTTP-Referer": "http://localhost",
        # "X-Title": "pdf-rag",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def answer_question(
    question: str,
    index_dir: str,
    llm_model: str,
    top_k: int = DEFAULT_TOP_K
) -> str:
    index, meta = load_index(index_dir)
    embed_model = meta["embed_model"]

    ctx = retrieve(question, index, meta, embed_model, top_k=top_k)

    context_blocks = []
    for j, item in enumerate(ctx, start=1):
        page = item["meta"].get("page")
        context_blocks.append(f"[Источник {j}, стр. {page}, score={item['score']:.3f}]\n{item['text']}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "Контекст не найден."

    system = (
        "Ты помощник, который отвечает на вопросы строго по предоставленному контексту из PDF. "
        "Если ответа нет в контексте — скажи, что в документе это не найдено. "
        "Цитируй страницы (стр. N), если возможно."
    )

    user = (
        f"Вопрос:\n{question}\n\n"
        f"Контекст из PDF:\n{context_text}\n\n"
        "Ответь по-русски. Если уместно, укажи ссылки на источники вида (стр. N)."
    )

    return openrouter_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=llm_model,
        temperature=0.2,
    )


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Построить индекс по PDF")
    b.add_argument("--pdf", default=DEFAULT_PDF_PATH)
    b.add_argument("--out", default=DEFAULT_INDEX_DIR)
    b.add_argument("--embed", default=DEFAULT_EMBED_MODEL)

    a = sub.add_parser("ask", help="Задать вопрос по PDF")
    a.add_argument("--index", default=DEFAULT_INDEX_DIR)
    a.add_argument("--model", default=DEFAULT_LLM_MODEL)
    a.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    a.add_argument("question", nargs="+", help="Текст вопроса")

    args = p.parse_args()

    if args.cmd == "build":
        build_index(args.pdf, args.out, args.embed)
        print(f"OK: индекс сохранён в {args.out}")
    elif args.cmd == "ask":
        q = " ".join(args.question)
        ans = answer_question(q, args.index, args.model, top_k=args.topk)
        print(ans)


if __name__ == "__main__":
    main()