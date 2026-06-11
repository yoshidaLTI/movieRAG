"""
proper_nouns_global.py

動画全体の OCR テキストから、全スライド共通の固有名詞リストを作成する（比較実験用）。

proper_nouns.py との違い:
  proper_nouns.py        → スライドごとに固有名詞を抽出（各スライドの proper_nouns フィールド）
  proper_nouns_global.py → 全スライドの OCR を一括送信し、共通固有名詞を抽出
                           → meta.proper_nouns_global フィールドに保存

使い方:
  python proper_nouns_global.py movie/test1/result.json
  python proper_nouns_global.py movie/test1/result.json --model gpt-oss-20b
"""

import json
import argparse
import re
from pathlib import Path

import requests

LMSTUDIO_URL  = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "gpt-oss-20b"

PROMPT = """\
以下は講義動画全体のスライドOCRテキストです。
Whisperの音声認識精度を上げるために、この講義全体を通じて繰り返し登場する固有名詞を
15個リストアップしてください。

抽出対象:
- 人名・組織名・地名/施設名
- 製品名・ツール名・フレームワーク名・ライブラリ名
- 専門用語・学術用語
- 略語・アクロニム

以下のJSON形式のみで返してください。説明・前置き・コードブロックは不要です。
{{"proper_nouns": ["用語1", "用語2", ...]}}

[OCRテキスト]
{ocr_text}"""


def collect_all_ocr(slides: list[dict]) -> str:
    parts = []
    for slide in slides:
        text = (slide.get("full_text", "") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_global_nouns(ocr_text: str, model: str) -> list[str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(ocr_text=ocr_text)}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    try:
        resp = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] LLM呼び出し失敗: {e}")
        return []

    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        data = json.loads(raw)
        nouns = data.get("proper_nouns", [])
        if isinstance(nouns, list):
            return [str(w) for w in nouns]
    except json.JSONDecodeError:
        print(f"[WARN] JSONパース失敗: {raw[:80]}")
    return []


def run(result_json_path: str, model: str):
    path = Path(result_json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    slides = data.get("slides", [])
    if not slides:
        print("[WARN] スライドが存在しません。")
        return

    print(f"[INFO] スライド数: {len(slides)}  モデル: {model}")

    ocr_text = collect_all_ocr(slides)
    print(f"[INFO] OCRテキスト総文字数: {len(ocr_text)}")
    print("[INFO] 固有名詞抽出中...")

    nouns = extract_global_nouns(ocr_text, model)
    print(f"[INFO] 抽出数: {len(nouns)}")
    print(f"[INFO] 内容: {nouns}")

    # meta フィールドに保存
    data.setdefault("meta", {})["proper_nouns_global"] = nouns

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {path}  →  meta.proper_nouns_global に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="全スライド共通の固有名詞リストを作成（比較実験用）")
    parser.add_argument("result_json", help="result.jsonのパス")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"LLMモデル名（デフォルト: {DEFAULT_MODEL}）")
    args = parser.parse_args()

    run(args.result_json, args.model)


if __name__ == "__main__":
    main()
