"""
rag_build.py

result.json から文字数チャンキングによる RAG インデックスを構築する。
3種類のインデックスを同時に作成する。

出力先（result.jsonと同じディレクトリ）:
  chroma_char_asr_ocr_mix/  ← OCR+ASR 混合
  chroma_char_ocr/          ← OCR のみ
  chroma_char_asr/          ← ASR のみ

使い方:
  python rag_build.py movie/test1/result.json
  python rag_build.py movie/test1/result.json --output-dir /path/to/out
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.indexer import build_index

DEFAULT_EMBEDDING_MODEL  = "cl-nagoya/ruri-v3-310m"
DEFAULT_MIN_ASR_CHARS    = 1000
DEFAULT_CHILD_CHUNK_SIZE = 300
DEFAULT_CHILD_OVERLAP    = 30
DEFAULT_USE_HINT         = True


def main():
    parser = argparse.ArgumentParser(
        description="result.json から文字数チャンキング RAG インデックスを構築",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result_json", help="result.jsonのパス")
    parser.add_argument("--output-dir", "-o",
                        help="出力先ディレクトリ（省略時: result.json と同じディレクトリ）")
    parser.add_argument("--embedding-model",     default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--min-asr-chars",       type=int, default=DEFAULT_MIN_ASR_CHARS)
    parser.add_argument("--child-chunk-size",    type=int, default=DEFAULT_CHILD_CHUNK_SIZE)
    parser.add_argument("--child-chunk-overlap", type=int, default=DEFAULT_CHILD_OVERLAP)
    parser.add_argument("--no-hint", action="store_true")
    args = parser.parse_args()

    result_json = Path(args.result_json)
    base_dir    = Path(args.output_dir) if args.output_dir else result_json.parent

    created = build_index(
        result_json_path    = result_json,
        base_dir            = base_dir,
        embedding_model     = args.embedding_model,
        min_asr_chars       = args.min_asr_chars,
        child_chunk_size    = args.child_chunk_size,
        child_chunk_overlap = args.child_chunk_overlap,
        use_hint            = not args.no_hint,
    )

    print("\n作成されたインデックス:")
    for mode, path in created.items():
        print(f"  [{mode}] {path}")
    print("\n検索例:")
    for path in created.values():
        print(f"  python rag_search.py \"クエリ\" --chroma {path}")


if __name__ == "__main__":
    main()
