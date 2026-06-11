# slide_base_rag/retriever.py

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import RuriEmbeddings
from rag.retriever import search_aligned  # noqa: F401  (re-export)


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
