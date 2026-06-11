"""
extract_asr.py

result.json から全スライドの ASR テキストを抽出し、テキストファイルに保存する。

出力フォーマット:
  [slide1 | 0.4s〜19.6s]
  テキスト...

  [slide2 | 19.6s〜155.1s]
  テキスト...

使い方:
  python extract_asr.py movie/test1/result.json
  python extract_asr.py movie/test1/result_no_prompt.json --field asr_no_prompt
  python extract_asr.py movie/test1/result.json --output asr.txt
"""

import json
import argparse
from pathlib import Path


def extract_asr(result_json_path: str, output_path: str | None, field: str):
    path = Path(result_json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    slides = data.get("slides", [])
    if not slides:
        print("[WARN] スライドが存在しません。")
        return

    lines = []
    for slide in slides:
        slide_id  = slide.get("slide_id", "")
        start_sec = slide.get("time", {}).get("start_sec", "")
        end_sec   = slide.get("time", {}).get("end_sec",   "")
        segments  = slide.get(field, [])

        time_str = f"{start_sec}s〜{end_sec}s" if start_sec != "" else ""
        header   = f"[{slide_id} | {time_str}]" if time_str else f"[{slide_id}]"
        lines.append(header)

        if segments:
            for seg in segments:
                text = seg.get("text", "").strip()
                if text:
                    lines.append(text)
        else:
            lines.append("（ASRなし）")

        lines.append("")

    output = "\n".join(lines).strip()

    if output_path:
        out_path = Path(output_path)
    else:
        suffix = field if field != "asr" else "asr"
        out_path = path.parent / f"{suffix}.txt"

    out_path.write_text(output, encoding="utf-8")
    print(f"[DONE] {out_path}  ({len(slides)}スライド)")


def main():
    parser = argparse.ArgumentParser(description="result.json から ASR テキストを抽出")
    parser.add_argument("result_json", help="result.jsonのパス")
    parser.add_argument("--field", "-f", default="asr",
                        help="抽出するフィールド名（デフォルト: asr、比較用: asr_no_prompt）")
    parser.add_argument("--output", "-o", help="出力ファイルパス（省略時: <field>.txt）")
    args = parser.parse_args()

    extract_asr(args.result_json, args.output, args.field)


if __name__ == "__main__":
    main()
