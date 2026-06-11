# qg/chunker.py

import json
from pathlib import Path


def load_result(result_json_path: str | Path) -> dict:
    with open(result_json_path, encoding="utf-8") as f:
        return json.load(f)


def get_ocr_text(slide: dict) -> str:
    return (slide.get("full_text", "") or "").strip()


def get_asr_text(slide: dict) -> str:
    segments = slide.get("asr", [])
    return "\n".join(s["text"] for s in segments if s.get("text", "").strip())


def chunk_by_slides(slides: list[dict], slides_per_chunk: int) -> list[dict]:
    """N枚ごとにスライドをチャンク分割"""
    chunks = []
    for i in range(0, len(slides), slides_per_chunk):
        group = slides[i:i + slides_per_chunk]
        chunks.append(_build_chunk(group, f"chunk_{i // slides_per_chunk:04d}"))
    return chunks


def chunk_by_time(slides: list[dict], minutes_per_chunk: float) -> list[dict]:
    """N分ごとにスライドをチャンク分割（チャンク先頭からの経過時間で区切る）"""
    threshold_sec = minutes_per_chunk * 60
    chunks = []
    buffer: list[dict] = []
    chunk_idx = 0

    for slide in slides:
        buffer.append(slide)
        start = buffer[0].get("time", {}).get("start_sec", 0.0)
        end   = slide.get("time", {}).get("end_sec", 0.0)
        if end - start >= threshold_sec:
            chunks.append(_build_chunk(buffer, f"chunk_{chunk_idx:04d}"))
            chunk_idx += 1
            buffer = []

    if buffer:
        chunks.append(_build_chunk(buffer, f"chunk_{chunk_idx:04d}"))

    return chunks


def _build_chunk(slides: list[dict], chunk_id: str) -> dict:
    ocr_parts: list[str] = []
    asr_parts: list[str] = []

    for slide in slides:
        ocr = get_ocr_text(slide)
        asr = get_asr_text(slide)
        if ocr:
            ocr_parts.append(f"[{slide['slide_id']}]\n{ocr}")
        if asr:
            asr_parts.append(asr)

    return {
        "chunk_id":  chunk_id,
        "slide_ids": [s["slide_id"] for s in slides],
        "start_sec": slides[0].get("time", {}).get("start_sec", 0.0),
        "end_sec":   slides[-1].get("time", {}).get("end_sec", 0.0),
        "ocr_text":  "\n\n".join(ocr_parts),
        "asr_text":  "\n".join(asr_parts),
    }
