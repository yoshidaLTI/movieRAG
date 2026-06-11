"""
qg_direct.py

result.json から直接質問を生成する（パターン1・2）。
結果は qg_result.json に追記される。

使い方:
  # パターン1: スライド枚数基準（3枚ごとに2問）
  python qg/qg_direct.py movie/lecture/result.json --by-slides --questions-per-chunk 2

  # パターン2: 経過時間基準（5分ごとに1問）
  python qg/qg_direct.py movie/lecture/result.json --by-time

  # チャンクサイズ・モデルを変更
  python qg/qg_direct.py movie/lecture/result.json --by-slides --slides-per-chunk 5 --model gpt-oss-20b
"""

import sys
import json
import re
import argparse
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from qg.chunker import load_result, chunk_by_slides, chunk_by_time

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

QG_PROMPT = """\
以下の講義テキスト（スライドのOCRと音声書き起こし）をもとに、講義内容を理解できているか確認するための4択問題を日本語で{n}問生成してください。

[講義テキスト]
{text}
{proper_nouns_section}
# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が [ で最後の文字が ] であること。

bloom_level は以下のいずれか: {bloom_levels}
answer は A, B, C, D のいずれか（正解の選択肢）

[{{"question": "質問文", "choice_A": "選択肢A", "choice_B": "選択肢B", "choice_C": "選択肢C", "choice_D": "選択肢D", "answer": "A", "reason": "正解の理由", "bloom_level": "知識"}}, ...]"""


def generate_questions(
    ocr_text: str,
    asr_text: str,
    n: int,
    model: str,
    bloom_level: str | None,
    proper_nouns: list[str],
    lmstudio_url: str = LMSTUDIO_URL,
) -> list[dict]:
    text_parts = []
    if ocr_text:
        text_parts.append(f"[スライド]\n{ocr_text}")
    if asr_text:
        text_parts.append(f"[音声]\n{asr_text}")
    text = "\n\n".join(text_parts)

    if not text.strip():
        return []

    proper_nouns_section = (
        f"[固有名詞・専門用語]\n{'、'.join(proper_nouns)}\n"
        if proper_nouns else ""
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": QG_PROMPT.format(
            n=n, text=text,
            proper_nouns_section=proper_nouns_section,
            bloom_levels=bloom_level or "知識、応用、評価",
        )}],
        "max_tokens": 2048 * n,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(lmstudio_url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] 質問生成失敗: {e}")
        return []

    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return [
                {
                    "question":    str(d.get("question", "")),
                    "choice_A":    str(d.get("choice_A", "")),
                    "choice_B":    str(d.get("choice_B", "")),
                    "choice_C":    str(d.get("choice_C", "")),
                    "choice_D":    str(d.get("choice_D", "")),
                    "answer":      str(d.get("answer", "")),
                    "reason":      str(d.get("reason", "")),
                    "bloom_level": str(d.get("bloom_level", "")),
                }
                for d in data
                if isinstance(d, dict) and d.get("question") and d.get("answer")
            ]
    except json.JSONDecodeError:
        pass
    return []


def run(
    result_json_path: str,
    chunk_mode: str,
    slides_per_chunk: int,
    minutes_per_chunk: float,
    questions_per_chunk: int,
    model: str,
    output_path: str | None,
    bloom_level: str | None,
    slide_overlap: int = 0,
    time_overlap_sec: float = 0.0,
    save_chunks: bool = False,
    lmstudio_url: str = LMSTUDIO_URL,
):
    data = load_result(result_json_path)
    slides = data.get("slides", [])
    proper_nouns = data.get("meta", {}).get("proper_nouns_global", [])

    if chunk_mode == "slides":
        chunks = chunk_by_slides(slides, slides_per_chunk, slide_overlap)
        mode_label = f"スライド{slides_per_chunk}枚単位 (overlap={slide_overlap}枚)"
        detail = {"chunk_mode": "slides", "slides_per_chunk": slides_per_chunk,
                  "slide_overlap": slide_overlap, "questions_per_chunk": questions_per_chunk,
                  "bloom_level": bloom_level}
    else:
        chunks = chunk_by_time(slides, minutes_per_chunk, time_overlap_sec)
        mode_label = f"{minutes_per_chunk}分単位 (overlap={time_overlap_sec}秒)"
        detail = {"chunk_mode": "time", "minutes_per_chunk": minutes_per_chunk,
                  "time_overlap_sec": time_overlap_sec, "questions_per_chunk": questions_per_chunk,
                  "bloom_level": bloom_level}

    print(f"[INFO] チャンク数: {len(chunks)}  モード: {mode_label}  1区間{questions_per_chunk}問  モデル: {model}")

    if save_chunks:
        chunks_path = Path(result_json_path).parent / f"qg_chunks_{chunk_mode}.json"
        chunk_log = [
            {
                "chunk_id":  c["chunk_id"],
                "slide_ids": c["slide_ids"],
                "start_sec": c["start_sec"],
                "end_sec":   c["end_sec"],
                "llm_input": {"ocr_text": c["ocr_text"], "asr_text": c["asr_text"]},
            }
            for c in chunks
        ]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_log, f, ensure_ascii=False, indent=2)
        print(f"[INFO] チャンク保存 → {chunks_path}")

    new_entries: list[dict] = []
    for chunk in chunks:
        label = f"{chunk['chunk_id']} ({chunk['start_sec']:.0f}s〜{chunk['end_sec']:.0f}s)"
        print(f"  {label} 生成中...", end=" ", flush=True)

        questions = generate_questions(
            chunk["ocr_text"], chunk["asr_text"], questions_per_chunk, model,
            bloom_level, proper_nouns, lmstudio_url,
        )
        if questions:
            for qi, q in enumerate(questions):
                qid = f"direct_{chunk_mode}_{chunk['chunk_id']}_q{qi:02d}"
                new_entries.append({
                    "question_id":    qid,
                    "model":          model,
                    "method":         "direct",
                    "detail_setting": detail,
                    "chunk_id":       chunk["chunk_id"],
                    "slide_ids":      chunk["slide_ids"],
                    "start_sec":      chunk["start_sec"],
                    "end_sec":        chunk["end_sec"],
                    "question":       q["question"],
                    "choice_A":       q["choice_A"],
                    "choice_B":       q["choice_B"],
                    "choice_C":       q["choice_C"],
                    "choice_D":       q["choice_D"],
                    "answer":         q["answer"],
                    "reason":         q["reason"],
                    "bloom_level":    q["bloom_level"],
                })
            print(f"{len(questions)}問 OK")
        else:
            print("SKIP")

    out_path = Path(output_path) if output_path else Path(result_json_path).parent / "qg_result.json"
    existing = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)

    all_entries = existing + new_entries
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {len(new_entries)}問追加（合計{len(all_entries)}問） → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="result.jsonから直接質問生成（パターン1・2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result", help="result.jsonのパス")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--by-slides", action="store_true", help="スライド枚数基準（パターン1）")
    mode_group.add_argument("--by-time",   action="store_true", help="経過時間基準（パターン2）")

    parser.add_argument("--slides-per-chunk",    type=int,   default=3,   help="スライド枚数基準のチャンクサイズ")
    parser.add_argument("--minutes-per-chunk",   type=float, default=5.0, help="経過時間基準のチャンクサイズ（分）")
    parser.add_argument("--slide-overlap",        type=int,   default=0,   help="スライド枚数基準の被り枚数（例: 1なら1枚重複）")
    parser.add_argument("--time-overlap",         type=float, default=0.0, help="経過時間基準の被り秒数（例: 60なら60秒重複）")
    parser.add_argument("--questions-per-chunk", type=int,   default=1,   help="1区間あたりの生成問題数")
    parser.add_argument("--bloom-level", default=None,
                        choices=["知識", "応用", "評価"],
                        help="ブルームレベルを1つに絞る（省略時: LLMが知識・応用・評価から選択）")
    parser.add_argument("--model",        default="gpt-oss-20b", help="LMStudioモデル名")
    parser.add_argument("--output",       default=None,                help="出力JSONパス（省略時: result.jsonと同ディレクトリのqg_result.json）")
    parser.add_argument("--save-chunks",  action="store_true",         help="LLMに渡すテキストをJSONに保存して確認する")
    parser.add_argument("--lmstudio-url", default=LMSTUDIO_URL,        help="LMStudioエンドポイント")

    args = parser.parse_args()
    chunk_mode = "slides" if args.by_slides else "time"

    run(
        result_json_path=args.result,
        chunk_mode=chunk_mode,
        slides_per_chunk=args.slides_per_chunk,
        minutes_per_chunk=args.minutes_per_chunk,
        questions_per_chunk=args.questions_per_chunk,
        model=args.model,
        output_path=args.output,
        bloom_level=args.bloom_level,
        slide_overlap=args.slide_overlap,
        time_overlap_sec=args.time_overlap,
        save_chunks=args.save_chunks,
        lmstudio_url=args.lmstudio_url,
    )


if __name__ == "__main__":
    main()
