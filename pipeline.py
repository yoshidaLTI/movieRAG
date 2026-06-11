"""
pipeline.py

slide_checker.py → proper_nouns_global.py → asr.py を順番に実行するラッパー。

使い方:
  python pipeline.py movie/test1.mp4
  python pipeline.py movie/test1.mp4 --output movie/test1
  python pipeline.py movie/test1.mp4 \\
      --ocr-model glm-ocr \\
      --nouns-model gpt-oss-20b \\
      --asr-model mlx-community/whisper-large-v3-mlx

途中のステップをスキップ:
  python pipeline.py movie/test1.mp4 --skip-ocr
  python pipeline.py movie/test1.mp4 --skip-ocr --skip-nouns
"""

import sys
import subprocess
import argparse
from pathlib import Path

# ────────────────────────────────────────────
# デフォルトモデル名
# ────────────────────────────────────────────
DEFAULT_OCR_MODEL   = "glm-ocr"
DEFAULT_NOUNS_MODEL = "gpt-oss-20b"
DEFAULT_ASR_MODEL   = "mlx-community/whisper-large-v3-mlx"


# ────────────────────────────────────────────
# ステップ実行
# ────────────────────────────────────────────
def run_step(cmd: list[str], label: str):
    print(f"\n{'━' * 55}")
    print(f"  {label}")
    print(f"{'━' * 55}")
    print(f"  $ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="スライド動画処理パイプライン（OCR → 固有名詞抽出 → ASR）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="動画ファイルパス (例: movie/test1.mp4)")
    parser.add_argument("--output", "-o",
                        help="出力ディレクトリ（省略時: <動画の親>/<動画stem>/）")

    # モデル指定
    model_group = parser.add_argument_group("モデル設定")
    model_group.add_argument("--ocr-model",   default=DEFAULT_OCR_MODEL,
                             help="OCRモデル名")
    model_group.add_argument("--nouns-model", default=DEFAULT_NOUNS_MODEL,
                             help="固有名詞抽出モデル名（動画全体の共通リストを作成）")
    model_group.add_argument("--asr-model",   default=DEFAULT_ASR_MODEL,
                             help="ASRモデル名")

    # スキップ制御
    skip_group = parser.add_argument_group("ステップスキップ")
    skip_group.add_argument("--skip-ocr",   action="store_true",
                            help="Step1 スライド検知&OCR をスキップ")
    skip_group.add_argument("--skip-nouns", action="store_true",
                            help="Step2 固有名詞抽出 をスキップ")
    skip_group.add_argument("--skip-asr",   action="store_true",
                            help="Step3 ASR をスキップ")

    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"[ERROR] 動画ファイルが見つかりません: {video}")
        sys.exit(1)

    output_dir  = Path(args.output) if args.output else video.parent / video.stem
    result_json = output_dir / "result.json"
    py          = sys.executable

    # ── Step 1: スライド検知 & OCR ──────────────
    if not args.skip_ocr:
        cmd = [py, "slide_checker.py", str(video),
               "--output", str(output_dir),
               "--model",  args.ocr_model]
        run_step(cmd, "Step 1 / 3  スライド検知 & OCR")
    else:
        print(f"\n[SKIP] Step 1: スライド検知&OCR")

    if not result_json.exists():
        print(f"[ERROR] result.json が存在しません: {result_json}")
        print("        --skip-ocr を使う場合は事前に slide_checker.py を実行してください。")
        sys.exit(1)

    # ── Step 2: 固有名詞抽出（動画全体共通） ────
    if not args.skip_nouns:
        cmd = [py, "proper_nouns_global.py", str(result_json),
               "--model", args.nouns_model]
        run_step(cmd, "Step 2 / 3  固有名詞抽出（動画全体共通）")
    else:
        print(f"[SKIP] Step 2: 固有名詞抽出")

    # ── Step 3: ASR ─────────────────────────────
    if not args.skip_asr:
        cmd = [py, "asr.py", str(video),
               "--result", str(result_json),
               "--model",  args.asr_model]
        run_step(cmd, "Step 3 / 3  音声認識 (ASR)")
    else:
        print(f"[SKIP] Step 3: ASR")

    print(f"\n{'━' * 55}")
    print(f"  [完了]  {result_json}")
    print(f"{'━' * 55}\n")


if __name__ == "__main__":
    main()
