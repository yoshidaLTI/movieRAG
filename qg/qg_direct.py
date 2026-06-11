"""
qg_direct.py

result.json から直接質問を生成する（パターン1・2）。

使い方:
  # パターン1: スライド枚数基準（3枚ごとに1問）
  python qg/qg_direct.py movie/lecture/result.json --by-slides

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
以下の講義テキスト（スライドのOCRと音声書き起こし）をもとに、講義内容を理解できているか確認するための質問と解答を1問生成してください。

[講義テキスト]
{text}

# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が {{ で最後の文字が }} であること。
{{"question": "質問文", "answer": "解答文"}}"""


def generate_question(
    ocr_text: str,
    asr_text: str,
    model: str,
    lmstudio_url: str = LMSTUDIO_URL,
) -> dict | None:
    text_parts = []
    if ocr_text:
        text_parts.append(f"[スライド]\n{ocr_text}")
    if asr_text:
        text_parts.append(f"[音声]\n{asr_text}")
    text = "\n\n".join(text_parts)

    if not text.strip():
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": QG_PROMPT.format(text=text)}],
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(lmstudio_url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] 質問生成失敗: {e}")
        return None

    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        data = json.loads(raw)
        if "question" in data and "answer" in data:
            return {"question": str(data["question"]), "answer": str(data["answer"])}
    except json.JSONDecodeError:
        pass
    return None


def run(
    result_json_path: str,
    chunk_mode: str,
    slides_per_chunk: int,
    minutes_per_chunk: float,
    model: str,
    output_path: str | None,
    lmstudio_url: str = LMSTUDIO_URL,
):
    data = load_result(result_json_path)
    slides = data.get("slides", [])

    if chunk_mode == "slides":
        chunks = chunk_by_slides(slides, slides_per_chunk)
        mode_label = f"スライド{slides_per_chunk}枚単位"
    else:
        chunks = chunk_by_time(slides, minutes_per_chunk)
        mode_label = f"{minutes_per_chunk}分単位"

    print(f"[INFO] チャンク数: {len(chunks)}  モード: {mode_label}  モデル: {model}")

    results = []
    for chunk in chunks:
        label = f"{chunk['chunk_id']} ({chunk['start_sec']:.0f}s〜{chunk['end_sec']:.0f}s)"
        print(f"  {label} 生成中...", end=" ", flush=True)

        qa = generate_question(chunk["ocr_text"], chunk["asr_text"], model, lmstudio_url)
        if qa:
            results.append({
                "chunk_id":  chunk["chunk_id"],
                "slide_ids": chunk["slide_ids"],
                "start_sec": chunk["start_sec"],
                "end_sec":   chunk["end_sec"],
                "question":  qa["question"],
                "answer":    qa["answer"],
            })
            print("OK")
        else:
            print("SKIP")

    out_path = (
        Path(output_path) if output_path
        else Path(result_json_path).parent / f"qg_direct_{chunk_mode}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {len(results)}問生成 → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="result.jsonから直接質問生成（パターン1・2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result", help="result.jsonのパス")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--by-slides", action="store_true", help="スライド枚数基準（パターン1）")
    mode_group.add_argument("--by-time",   action="store_true", help="経過時間基準（パターン2）")

    parser.add_argument("--slides-per-chunk",  type=int,   default=3,   help="スライド枚数基準のチャンクサイズ")
    parser.add_argument("--minutes-per-chunk", type=float, default=5.0, help="経過時間基準のチャンクサイズ（分）")
    parser.add_argument("--model",  default="qwen/qwen3-vl-8b", help="LMStudioモデル名")
    parser.add_argument("--output", default=None,                help="出力JSONパス（省略時: result.jsonと同ディレクトリ）")
    parser.add_argument("--lmstudio-url", default=LMSTUDIO_URL,  help="LMStudioエンドポイント")

    args = parser.parse_args()
    chunk_mode = "slides" if args.by_slides else "time"

    run(
        result_json_path=args.result,
        chunk_mode=chunk_mode,
        slides_per_chunk=args.slides_per_chunk,
        minutes_per_chunk=args.minutes_per_chunk,
        model=args.model,
        output_path=args.output,
        lmstudio_url=args.lmstudio_url,
    )


if __name__ == "__main__":
    main()
