# slide_base_rag/indexer.py

import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag.embeddings import RuriEmbeddings


# ────────────────────────────────────────────
# テキスト正規化
# ────────────────────────────────────────────
def normalize_text(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    deduped = []
    for line in lines:
        stripped = line.strip()
        if stripped and (not deduped or stripped != deduped[-1].strip()):
            deduped.append(line)
    return "\n".join(deduped).strip()


# ────────────────────────────────────────────
# フレームごとの OCR / ASR テキストを取得
# ────────────────────────────────────────────
def get_frame_segments(slide: dict) -> list[dict]:
    """
    スライド内の各フレームについて
    { ocr_text, asr_text, start_sec, end_sec, frame_id, mode } を返す。

    フレームの時間範囲: detect_time 〜 次フレームの detect_time（最後はスライド終端）
    ASR: その時間範囲に含まれるセグメントを結合
    """
    frames    = slide.get("frames", [])
    asr_segs  = slide.get("asr", [])
    slide_end = slide.get("time", {}).get("end_sec", 0.0)

    result = []
    for i, frame in enumerate(frames):
        start = frame.get("detect_time", frame.get("time", 0.0))
        end   = (
            frames[i + 1].get("detect_time", frames[i + 1].get("time", 0.0))
            if i + 1 < len(frames)
            else slide_end
        )

        asr_texts = [
            s["text"].strip()
            for s in asr_segs
            if s.get("text", "").strip() and start <= s["start_sec"] < end
        ]

        result.append({
            "ocr_text":  normalize_text(frame.get("ocr", "") or ""),
            "asr_text":  normalize_text("\n".join(asr_texts)),
            "start_sec": start,
            "end_sec":   end,
            "frame_id":  frame.get("id", ""),
            "mode":      frame.get("mode", ""),
        })

    return result


# ────────────────────────────────────────────
# 親テキスト構築（スライドグループ単位）
# ────────────────────────────────────────────
def build_group_texts(slides: list[dict]) -> dict:
    """グループ内全スライドの combined / ocr / asr テキストをスライド境界付きで返す。"""
    combined_parts, ocr_parts, asr_parts = [], [], []

    for slide in slides:
        sid = slide.get("slide_id", "")
        ocr = normalize_text(slide.get("full_text", "") or "")
        asr_texts = [
            s["text"].strip()
            for s in slide.get("asr", [])
            if s.get("text", "").strip()
        ]
        asr = normalize_text("\n".join(asr_texts))

        # combined: スライド単位で OCR と ASR を対応付けて格納
        if ocr or asr:
            block = f"[{sid}]"
            if ocr: block += f"\n{ocr}"
            if asr: block += f"\n[音声]\n{asr}"
            combined_parts.append(block)

        # ocr / asr 単独: スライド ID を見出しとして付ける
        if ocr:
            ocr_parts.append(f"[{sid}]\n{ocr}")
        if asr:
            asr_parts.append(f"[{sid}]\n{asr}")

    return {
        "combined": "\n\n".join(combined_parts),
        "ocr":      "\n\n".join(ocr_parts),
        "asr":      "\n\n".join(asr_parts),
    }


# ────────────────────────────────────────────
# Document 構築（3種類まとめて）
# ────────────────────────────────────────────
def build_all_documents(
    slides: list[dict],
    group_size: int,
    use_hint: bool,
) -> dict[str, tuple[list[Document], list[Document]]]:
    """
    戻り値:
      {
        "combined": (parent_docs, child_docs),
        "ocr":      (parent_docs, child_docs),
        "asr":      (parent_docs, child_docs),
      }
    """
    docs: dict[str, tuple] = {
        "combined": ([], []),
        "ocr":      ([], []),
        "asr":      ([], []),
    }

    groups = [slides[i:i + group_size] for i in range(0, len(slides), group_size)]

    for g_idx, group in enumerate(groups):
        chunk_id  = f"parent_{g_idx:04d}"
        slide_ids = [s["slide_id"] for s in group]
        start_sec = group[0].get("time", {}).get("start_sec", 0.0)
        end_sec   = group[-1].get("time", {}).get("end_sec",   0.0)
        image     = group[0]["frames"][0].get("image", "") if group[0].get("frames") else ""

        base_meta = {
            "chunk_id":  chunk_id,
            "slide_ids": ",".join(slide_ids),
            "start_sec": start_sec,
            "end_sec":   end_sec,
            "image":     image,
        }

        # 親テキストを3種まとめて生成
        texts = build_group_texts(group)

        for mode in ("combined", "ocr", "asr"):
            parent_text = texts[mode]
            if not parent_text:
                continue
            docs[mode][0].append(Document(
                page_content=parent_text,
                metadata={**base_meta, "store_mode": mode},
            ))

        # 子チャンク: グループ内の全フレーム × 3種
        child_idx = 0
        for slide in group:
            hints      = slide.get("proper_nouns", []) if use_hint else []
            hint_block = f"[キーワード]\n{'、'.join(hints)}\n\n" if hints else ""

            for seg in get_frame_segments(slide):
                ocr = seg["ocr_text"]
                asr = seg["asr_text"]

                combined_parts = []
                if ocr: combined_parts.append(ocr)
                if asr: combined_parts.append(f"[音声]\n{asr}")
                combined_text = "\n\n".join(combined_parts)

                child_meta = {
                    "chunk_id":    chunk_id,
                    "child_index": child_idx,
                    "slide_id":    slide["slide_id"],
                    "frame_id":    seg["frame_id"],
                    "mode":        seg["mode"],
                    "start_sec":   seg["start_sec"],
                    "end_sec":     seg["end_sec"],
                }

                for store_mode, content in [
                    ("combined", hint_block + combined_text),
                    ("ocr",      hint_block + ocr),
                    ("asr",      hint_block + asr),
                ]:
                    if content.strip():
                        docs[store_mode][1].append(Document(
                            page_content=content,
                            metadata={**child_meta, "store_mode": store_mode},
                        ))

                child_idx += 1

    return docs


# ────────────────────────────────────────────
# ChromaDB への登録
# ────────────────────────────────────────────
CHROMA_DIR_MAP = {
    "combined": "chroma_slide_asr_ocr_mix",
    "ocr":      "chroma_slide_ocr",
    "asr":      "chroma_slide_asr",
}


def build_index(
    result_json_path: str | Path,
    base_dir: str | Path | None = None,
    embedding_model: str        = "cl-nagoya/ruri-v3-310m",
    group_size: int             = 3,
    use_hint: bool              = True,
) -> dict[str, Path]:
    """
    combined / ocr / asr の3種類を同時に構築する。
    各モードを独立したディレクトリに保存する。

      <base_dir>/chroma_slide_asr_ocr_mix/  ← OCR+ASR 混合
      <base_dir>/chroma_slide_ocr/          ← OCR のみ
      <base_dir>/chroma_slide_asr/          ← ASR のみ

    base_dir 省略時は result_json_path の親ディレクトリを使用。
    戻り値: {"combined": Path, "ocr": Path, "asr": Path}
    """
    result_json_path = Path(result_json_path)
    base_dir         = Path(base_dir) if base_dir else result_json_path.parent

    print(f"[indexer] result.json 読み込み: {result_json_path}")
    with open(result_json_path, encoding="utf-8") as f:
        data = json.load(f)

    slides = data.get("slides", [])
    print(f"[indexer] スライド数: {len(slides)}  グループサイズ: {group_size}")

    all_docs = build_all_documents(slides, group_size, use_hint)
    for mode, (parent_docs, child_docs) in all_docs.items():
        print(f"[indexer] [{mode}] 親: {len(parent_docs)}  子: {len(child_docs)}")

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
