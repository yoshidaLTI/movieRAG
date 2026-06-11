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
以下の講義テキスト（スライドのOCRと音声書き起こし）をもとに、講義内容を理解できているか確認するための質問と解答を{n}問生成してください。

[講義テキスト]
{text}

# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が [ で最後の文字が ] であること。

question_type は以下のいずれか: 一問一答, 多岐選択問題, 記述型
bloom_level は以下のいずれか: 知識, 応用, 評価

[{{"question": "質問文", "answer": "解答文", "question_type": "一問一答", "bloom_level": "知識"}}, ...]"""


def generate_questions(
    ocr_text: str,
    asr_text: str,
    n: int,
    model: str,
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

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": QG_PROMPT.format(n=n, text=text)}],
        "max_tokens": 512 * n,
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
                    "question":      str(d.get("question", "")),
                    "answer":        str(d.get("answer", "")),
                    "question_type": str(d.get("question_type", "")),
                    "bloom_level":   str(d.get("bloom_level", "")),
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
    save_chunks: bool = False,
    lmstudio_url: str = LMSTUDIO_URL,
):
    data = load_result(result_json_path)
    slides = data.get("slides", [])

    if chunk_mode == "slides":
        chunks = chunk_by_slides(slides, slides_per_chunk)
        mode_label = f"スライド{slides_per_chunk}枚単位"
        detail = {"chunk_mode": "slides", "slides_per_chunk": slides_per_chunk,
                  "questions_per_chunk": questions_per_chunk}
    else:
        chunks = chunk_by_time(slides, minutes_per_chunk)
        mode_label = f"{minutes_per_chunk}分単位"
        detail = {"chunk_mode": "time", "minutes_per_chunk": minutes_per_chunk,
                  "questions_per_chunk": questions_per_chunk}

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
            chunk["ocr_text"], chunk["asr_text"], questions_per_chunk, model, lmstudio_url
        )
        if questions:
            for qi, q in enumerate(questions):
                qid = f"direct_{chunk_mode}_{chunk['chunk_id']}_q{qi:02d}"
                new_entries.append({
                    "question_id":   qid,
                    "model":         model,
                    "method":        "direct",
                    "detail_setting": detail,
                    "chunk_id":      chunk["chunk_id"],
                    "slide_ids":     chunk["slide_ids"],
                    "start_sec":     chunk["start_sec"],
                    "end_sec":       chunk["end_sec"],
                    "question":      q["question"],
                    "answer":        q["answer"],
                    "question_type": q["question_type"],
                    "bloom_level":   q["bloom_level"],
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
    parser.add_argument("--questions-per-chunk", type=int,   default=1,   help="1区間あたりの生成問題数")
    parser.add_argument("--model",        default="qwen/qwen3-vl-8b", help="LMStudioモデル名")
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
        save_chunks=args.save_chunks,
        lmstudio_url=args.lmstudio_url,
    )


if __name__ == "__main__":
    main()
