# rag/retriever.py

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import RuriEmbeddings


def load_stores(
    chroma_dir: str | Path,
    embedding_model: str = "cl-nagoya/ruri-v3-310m",
) -> tuple[Chroma, Chroma]:
    embeddings = RuriEmbeddings(embedding_model)
    chroma_dir = str(chroma_dir)
    child_store  = Chroma(collection_name="child_chunks",
                          embedding_function=embeddings,
                          persist_directory=chroma_dir)
    parent_store = Chroma(collection_name="parent_chunks",
                          embedding_function=embeddings,
                          persist_directory=chroma_dir)
    return child_store, parent_store


def search(
    query: str,
    child_store: Chroma,
    parent_store: Chroma,
    k: int = 3,
) -> list[Document]:
    child_hits = child_store.similarity_search(query, k=k * 2)

    seen, chunk_ids = set(), []
    for doc in child_hits:
        cid = doc.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            chunk_ids.append(cid)
            if len(chunk_ids) >= k:
                break

    parent_docs = []
    for cid in chunk_ids:
        results = parent_store.get(ids=[cid], include=["documents", "metadatas"])
        if results["documents"]:
            parent_docs.append(Document(
                page_content=results["documents"][0],
                metadata=results["metadatas"][0],
            ))
    return parent_docs


# ────────────────────────────────────────────
# 時間的整合あり検索
# ASRで上位k/2件 → 同chunk_idのOCRを逆引き
# OCRで上位k/2件 → 同chunk_idのASRを逆引き
# ────────────────────────────────────────────

def _top_chunk_ids(child_store: Chroma, query: str, k: int) -> list[str]:
    hits = child_store.similarity_search(query, k=k * 2)
    seen, ids = set(), []
    for doc in hits:
        cid = doc.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
            if len(ids) >= k:
                break
    return ids


def _fetch_parent(parent_store: Chroma, chunk_id: str) -> Document | None:
    results = parent_store.get(ids=[chunk_id], include=["documents", "metadatas"])
    if results["documents"]:
        return Document(
            page_content=results["documents"][0],
            metadata=results["metadatas"][0],
        )
    return None


def search_aligned(
    query: str,
    asr_child: Chroma,
    asr_parent: Chroma,
    ocr_child: Chroma,
    ocr_parent: Chroma,
    k: int = 3,
) -> list[Document]:
    """
    ASR DBで上位k/2件を検索し同chunk_idのOCRを逆引き、
    OCR DBで上位k/2件を検索し同chunk_idのASRを逆引きして結合する。
    LLMへ渡すコンテキスト量は通常のsearch()と同じ(OCR+ASR)*k件相当。
    """
    half = max(1, k // 2)

    asr_ids = _top_chunk_ids(asr_child, query, half)
    ocr_ids = _top_chunk_ids(ocr_child, query, half)

    seen: set[str] = set()
    ordered_ids: list[str] = []
    for cid in asr_ids + ocr_ids:
        if cid not in seen:
            seen.add(cid)
            ordered_ids.append(cid)

    results = []
    for cid in ordered_ids[:k]:
        ocr_doc = _fetch_parent(ocr_parent, cid)
        asr_doc = _fetch_parent(asr_parent, cid)

        parts = []
        meta = {}
        if ocr_doc:
            parts.append(ocr_doc.page_content)
            meta = ocr_doc.metadata
        if asr_doc:
            parts.append(f"[音声]\n{asr_doc.page_content}")
            if not meta:
                meta = asr_doc.metadata

        if parts:
            results.append(Document(
                page_content="\n\n".join(parts),
                metadata=meta,
            ))

    return results
