"""
rag_build_slide.py

result.json からスライド単位 RAG インデックスを構築する。
3種類のインデックスを同時に作成する。

出力先（result.jsonと同じディレクトリ）:
  chroma_slide_asr_ocr_mix/  ← OCR+ASR 混合
  chroma_slide_ocr/          ← OCR のみ
  chroma_slide_asr/          ← ASR のみ

使い方:
  python rag_build_slide.py movie/test1/result.json
  python rag_build_slide.py movie/test1/result.json --group-size 3
  python rag_build_slide.py movie/test1/result.json --output-dir /path/to/out
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slide_base_rag.indexer import build_index

DEFAULT_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"
DEFAULT_GROUP_SIZE      = 3


def main():
    parser = argparse.ArgumentParser(
        description="result.json からスライド単位 RAG インデックスを構築",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result_json", help="result.jsonのパス")
    parser.add_argument("--output-dir", "-o",
                        help="出力先ディレクトリ（省略時: result.json と同じディレクトリ）")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                        help="親チャンクにまとめるスライド枚数")
    parser.add_argument("--no-hint", action="store_true")
    args = parser.parse_args()

    result_json = Path(args.result_json)
    base_dir    = Path(args.output_dir) if args.output_dir else result_json.parent

    created = build_index(
        result_json_path = result_json,
        base_dir         = base_dir,
        embedding_model  = args.embedding_model,
        group_size       = args.group_size,
        use_hint         = not args.no_hint,
    )

    print("\n作成されたインデックス:")
    for mode, path in created.items():
        print(f"  [{mode}] {path}")
    print("\n検索例:")
    for path in created.values():
        print(f"  python rag_search_slide.py \"クエリ\" --chroma {path}")


if __name__ == "__main__":
    main()
