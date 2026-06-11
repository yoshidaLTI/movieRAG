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


def chunk_by_slides(
    slides: list[dict],
    slides_per_chunk: int,
    slide_overlap: int = 0,
) -> list[dict]:
    """N枚ごとにスライドをチャンク分割。slide_overlap枚だけ隣のチャンクと重複させる。"""
    step = max(1, slides_per_chunk - slide_overlap)
    return [
        _build_chunk(slides[i:i + slides_per_chunk], f"chunk_{chunk_idx:04d}")
        for chunk_idx, i in enumerate(range(0, len(slides), step))
    ]


def chunk_by_time(
    slides: list[dict],
    minutes_per_chunk: float,
    time_overlap_sec: float = 0.0,
) -> list[dict]:
    """N分ごとにスライドをチャンク分割。time_overlap_sec秒だけ隣のチャンクと重複させる。"""
    threshold_sec = minutes_per_chunk * 60
    chunks = []
    chunk_idx = 0
    start_i = 0

    while start_i < len(slides):
        buffer = []
        hit_threshold = False
        last_j = len(slides) - 1

        for j in range(start_i, len(slides)):
            buffer.append(slides[j])
            chunk_start = slides[start_i].get("time", {}).get("start_sec", 0.0)
            chunk_end = slides[j].get("time", {}).get("end_sec", 0.0)
            if chunk_end - chunk_start >= threshold_sec:
                hit_threshold = True
                last_j = j
                break

        chunks.append(_build_chunk(buffer, f"chunk_{chunk_idx:04d}"))
        chunk_idx += 1

        if not hit_threshold:
            break

        chunk_end_time = slides[last_j].get("time", {}).get("end_sec", 0.0)
        next_anchor = chunk_end_time - time_overlap_sec
        next_i = last_j + 1

        if time_overlap_sec > 0:
            for k in range(last_j, start_i, -1):
                if slides[k].get("time", {}).get("start_sec", 0.0) >= next_anchor:
                    next_i = k
                else:
                    break

        start_i = max(start_i + 1, next_i)

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
