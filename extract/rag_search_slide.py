"""
rag_search_slide.py

スライド単位RAGインデックスに対してクエリを実行する。

使い方:
  python rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_asr_ocr_mix
  python rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_ocr
  python rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_asr --k 5

時間的整合ありモード（_asr/_ocr ペアを相互逆引き）:
  python rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_asr --aligned
  python rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_ocr --aligned
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slide_base_rag.retriever import load_stores, search, search_aligned


def _sibling_chroma(chroma_path: Path) -> Path:
    name = chroma_path.name
    if name.endswith("_asr"):
        return chroma_path.parent / (name[:-3] + "ocr")
    if name.endswith("_ocr"):
        return chroma_path.parent / (name[:-3] + "asr")
    raise ValueError(
        f"--aligned には _asr または _ocr で終わるディレクトリが必要です: {chroma_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="スライド単位RAG 検索")
    parser.add_argument("query", help="検索クエリ")
    parser.add_argument("--chroma", "-c", required=True, help="ChromaDB ディレクトリ")
    parser.add_argument("--k", type=int, default=3, help="返す親チャンク数")
    parser.add_argument("--embedding-model", default="cl-nagoya/ruri-v3-310m")
    parser.add_argument(
        "--aligned", action="store_true",
        help="時間的整合ありモード: ASR/OCR を相互逆引きして結合（_asr/_ocr ペアが必要）",
    )
    args = parser.parse_args()

    if args.aligned:
        chroma_path = Path(args.chroma)
        sibling_path = _sibling_chroma(chroma_path)

        if chroma_path.name.endswith("_asr"):
            asr_path, ocr_path = chroma_path, sibling_path
        else:
            asr_path, ocr_path = sibling_path, chroma_path

        asr_child, asr_parent = load_stores(asr_path, args.embedding_model)
        ocr_child, ocr_parent = load_stores(ocr_path, args.embedding_model)
        results = search_aligned(
            args.query, asr_child, asr_parent, ocr_child, ocr_parent, k=args.k
        )
        mode_label = "aligned (ASR↔OCR 相互逆引き)"
    else:
        child_store, parent_store = load_stores(args.chroma, args.embedding_model)
        results = search(args.query, child_store, parent_store, k=args.k)
        mode_label = args.chroma

    print(f"\nクエリ: 「{args.query}」  mode: {mode_label}\n")
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"{'─'*60}")
        print(f"[{i}] {m.get('slide_ids')}  {m.get('start_sec')}s〜{m.get('end_sec')}s")
        print(doc.page_content[:500])
        if len(doc.page_content) > 500:
            print("  ...")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
