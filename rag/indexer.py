# rag/indexer.py

import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import RuriEmbeddings


# ────────────────────────────────────────────
# テキスト正規化
# ────────────────────────────────────────────
def normalize_text(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    # 空行・重複行を除去
    deduped = []
    for line in lines:
        stripped = line.strip()
        if stripped and (not deduped or stripped != deduped[-1].strip()):
            deduped.append(line)
    return "\n".join(deduped).strip()


# ────────────────────────────────────────────
# result.json からのテキスト抽出
# ────────────────────────────────────────────
def get_ocr_text(slide: dict) -> str:
    """スライドの最終フレームOCRテキストを取得（full_text フィールド）"""
    return normalize_text(slide.get("full_text", "") or "")


def get_asr_text(slide: dict) -> str:
    """スライドのASRセグメントを結合して取得"""
    segments = slide.get("asr", [])
    text = "\n".join(s["text"] for s in segments if s.get("text", "").strip())
    return normalize_text(text)


def get_hint_text(slide: dict) -> str:
    """proper_nouns をキーワードヒントとして取得"""
    nouns = slide.get("proper_nouns", [])
    return "、".join(nouns) if nouns else ""


# ────────────────────────────────────────────
# 親チャンクの構築
# ────────────────────────────────────────────
def build_parent_chunks(slides: list[dict], min_asr_chars: int) -> list[dict]:
    """
    ASR文字数が min_asr_chars に達するまでスライドを結合して親チャンクを作る。
    文字数が足りない末尾スライドは最後の親チャンクに追記する。
    """
    parent_chunks = []
    buffer: list[dict] = []
    buffer_asr_chars = 0

    for slide in slides:
        asr = get_asr_text(slide)
        buffer.append(slide)
        buffer_asr_chars += len(asr)

        if buffer_asr_chars >= min_asr_chars:
            parent_chunks.append(_merge_slides(buffer))
            buffer = []
            buffer_asr_chars = 0

    # 残りスライドの処理
    if buffer:
        if parent_chunks:
            last = parent_chunks[-1]
            last["slide_ids"] += [s["slide_id"] for s in buffer]
            last["end_sec"] = _slide_end_sec(buffer[-1])
            for slide in buffer:
                ocr  = get_ocr_text(slide)
                asr  = get_asr_text(slide)
                hint = get_hint_text(slide)
                if ocr:  last["ocr_text"] += f"\n\n{ocr}"
                if asr:  last["asr_text"] += f"\n{asr}"
                if hint: last["hints"].append(hint)
            last["parent_text"] = _build_parent_text(last["ocr_text"], last["asr_text"])
        else:
            parent_chunks.append(_merge_slides(buffer))

    return parent_chunks


def _slide_end_sec(slide: dict) -> float:
    return slide.get("time", {}).get("end_sec", 0.0)


def _build_parent_text(ocr: str, asr: str) -> str:
    parts = []
    if ocr: parts.append(ocr)
    if asr: parts.append(f"[音声]\n{asr}")
    return "\n\n".join(parts)


def _merge_slides(slides: list[dict]) -> dict:
    ocr_parts, asr_parts, hints = [], [], []

    for slide in slides:
        ocr  = get_ocr_text(slide)
        asr  = get_asr_text(slide)
        hint = get_hint_text(slide)
        if ocr:  ocr_parts.append(ocr)
        if asr:  asr_parts.append(asr)
        if hint: hints.append(hint)

    ocr_text = "\n\n".join(ocr_parts)
    asr_text = "\n".join(asr_parts)

    return {
        "slide_ids":   [s["slide_id"] for s in slides],
        "start_sec":   slides[0].get("time", {}).get("start_sec", 0.0),
        "end_sec":     _slide_end_sec(slides[-1]),
        "image":       slides[0]["frames"][0]["image"] if slides[0].get("frames") else "",
        "parent_text": _build_parent_text(ocr_text, asr_text),
        "ocr_text":    ocr_text,
        "asr_text":    asr_text,
        "hints":       hints,
    }


# ────────────────────────────────────────────
# LangChain Document への変換（3種類まとめて）
# ────────────────────────────────────────────
def to_all_documents(
    parent_chunks: list[dict],
    child_chunk_size: int,
    child_chunk_overlap: int,
    use_hint: bool,
) -> dict[str, tuple[list[Document], list[Document]]]:
    """
    combined / ocr / asr の3種類の (parent_docs, child_docs) を返す。

    combined: OCR+ASR 全文を親に、ASR（なければOCR）を子に
    ocr:      OCR のみを親・子に
    asr:      ASR のみを親・子に
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )

    docs: dict[str, tuple] = {
        "combined": ([], []),
        "ocr":      ([], []),
        "asr":      ([], []),
    }

    for i, chunk in enumerate(parent_chunks):
        chunk_id  = f"parent_{i:04d}"
        base_meta = {
            "chunk_id":  chunk_id,
            "slide_ids": ",".join(chunk["slide_ids"]),
            "start_sec": chunk["start_sec"],
            "end_sec":   chunk["end_sec"],
            "image":     chunk["image"],
        }
        hint_block = ""
        if use_hint and chunk["hints"]:
            hint_block = f"[キーワード]\n{'、'.join(chunk['hints'])}\n\n"

        # ── 親ドキュメント ──
        for mode, parent_text in [
            ("combined", chunk["parent_text"]),
            ("ocr",      chunk["ocr_text"]),
            ("asr",      chunk["asr_text"]),
        ]:
            if parent_text:
                docs[mode][0].append(Document(
                    page_content=parent_text,
                    metadata={**base_meta, "store_mode": mode},
                ))

        # ── 子ドキュメント ──
        for mode, child_base_text in [
            ("combined", chunk["asr_text"] or chunk["ocr_text"]),
            ("ocr",      chunk["ocr_text"]),
            ("asr",      chunk["asr_text"]),
        ]:
            if not child_base_text:
                continue
            full_text = hint_block + child_base_text
            for j, text in enumerate(splitter.split_text(full_text)):
                docs[mode][1].append(Document(
                    page_content=text,
                    metadata={
                        "chunk_id":    chunk_id,
                        "child_index": j,
                        "slide_ids":   ",".join(chunk["slide_ids"]),
                        "start_sec":   chunk["start_sec"],
                        "store_mode":  mode,
                    },
                ))

    return docs


# ────────────────────────────────────────────
# ChromaDB への登録
# ────────────────────────────────────────────
CHROMA_DIR_MAP = {
    "combined": "chroma_char_asr_ocr_mix",
    "ocr":      "chroma_char_ocr",
    "asr":      "chroma_char_asr",
}


def build_index(
    result_json_path: str | Path,
    base_dir: str | Path | None  = None,
    embedding_model: str         = "cl-nagoya/ruri-v3-310m",
    min_asr_chars: int           = 1000,
    child_chunk_size: int        = 300,
    child_chunk_overlap: int     = 30,
    use_hint: bool               = True,
) -> dict[str, Path]:
    """
    combined / ocr / asr の3種類を同時に構築する。
    各モードを独立したディレクトリに保存する。

      <base_dir>/chroma_char_asr_ocr_mix/  ← OCR+ASR 混合
      <base_dir>/chroma_char_ocr/          ← OCR のみ
      <base_dir>/chroma_char_asr/          ← ASR のみ

    base_dir 省略時は result_json_path の親ディレクトリを使用。
    戻り値: {"combined": Path, "ocr": Path, "asr": Path}
    """
    result_json_path = Path(result_json_path)
    base_dir         = Path(base_dir) if base_dir else result_json_path.parent

    print(f"[indexer] result.json 読み込み: {result_json_path}")
    with open(result_json_path, encoding="utf-8") as f:
        data = json.load(f)

    slides = data.get("slides", [])
    print(f"[indexer] スライド数: {len(slides)}")

    parent_chunks = build_parent_chunks(slides, min_asr_chars)
    print(f"[indexer] 親チャンク数: {len(parent_chunks)}")

    all_docs = to_all_documents(
        parent_chunks, child_chunk_size, child_chunk_overlap, use_hint
    )
    for mode, (p, c) in all_docs.items():
        print(f"[indexer] [{mode}] 親: {len(p)}  子: {len(c)}")

    embeddings = RuriEmbeddings(embedding_model)

    def register(chroma_dir: Path, parent_docs, child_docs):
        chroma_dir.mkdir(parents=True, exist_ok=True)

        def fresh(name):
            s = Chroma(collection_name=name, embedding_function=embeddings,
                       persist_directory=str(chroma_dir))
            s.delete_collection()
            return Chroma(collection_name=name, embedding_function=embeddings,
                          persist_directory=str(chroma_dir))

        child_store  = fresh("child_chunks")
        parent_store = fresh("parent_chunks")
        child_ids  = [f"{d.metadata['chunk_id']}_child_{d.metadata['child_index']}"
                      for d in child_docs]
        parent_ids = [d.metadata["chunk_id"] for d in parent_docs]
        child_store.add_documents(child_docs,   ids=child_ids)
        parent_store.add_documents(parent_docs, ids=parent_ids)

    created = {}
    for mode, (parent_docs, child_docs) in all_docs.items():
        if not parent_docs:
            continue
        chroma_dir = base_dir / CHROMA_DIR_MAP[mode]
        register(chroma_dir, parent_docs, child_docs)
        print(f"[indexer] [{mode}] 登録完了 → {chroma_dir}")
        created[mode] = chroma_dir

    print(f"\n[indexer] 全コレクション登録完了")
    return created
