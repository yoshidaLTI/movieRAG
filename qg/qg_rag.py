"""
qg_rag.py

RAGを使って質問を生成する（パターン3・4）。

処理フロー:
  1. チャンクのOCR+ASRからキーワード（seed）をLLMで抽出
  2. seedでRAG類似度検索
  3. 検索結果コンテキスト＋seedから質問・解答を生成

使い方:
  # パターン3: スライド枚数基準（3枚ごとに2問）
  python qg/qg_rag.py movie/lecture/result.json \
      --chroma movie/lecture/chroma_char_asr_ocr_mix \
      --by-slides --questions-per-chunk 2

  # パターン4: 経過時間基準（5分ごとに1問）
  python qg/qg_rag.py movie/lecture/result.json \
      --chroma movie/lecture/chroma_char_asr_ocr_mix \
      --by-time

  # モデル・チャンクサイズを変更
  python qg/qg_rag.py movie/lecture/result.json \
      --chroma movie/lecture/chroma_slide_asr_ocr_mix \
      --by-slides --slides-per-chunk 5 --model gpt-oss-20b --rag-k 5
"""

import sys
import json
import re
import argparse
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from qg.chunker import load_result, chunk_by_slides, chunk_by_time
from qg.seed_extractor import extract_seeds
from rag.retriever import load_stores, search

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

QG_RAG_PROMPT = """\
以下の講義資料（RAGで検索した関連コンテキスト）とキーワードをもとに、講義内容を理解できているか確認するための質問と解答を{n}問生成してください。

[キーワード]
{seeds}

[講義資料]
{context}

# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が [ で最後の文字が ] であること。
[{{"question": "質問文", "answer": "解答文"}}, ...]"""


def generate_questions_with_rag(
    seeds: list[str],
    context: str,
    n: int,
    model: str,
    lmstudio_url: str = LMSTUDIO_URL,
) -> list[dict]:
    if not seeds or not context.strip():
        return []

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": QG_RAG_PROMPT.format(
            n=n,
            seeds="、".join(seeds),
            context=context,
        )}],
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
        if isinstance(data, list):
            return [
                {"question": str(d["question"]), "answer": str(d["answer"])}
                for d in data
                if isinstance(d, dict) and "question" in d and "answer" in d
            ]
        if isinstance(data, dict) and "question" in data and "answer" in data:
            return [{"question": str(data["question"]), "answer": str(data["answer"])}]
    except json.JSONDecodeError:
        pass
    return []


def run(
    result_json_path: str,
    chroma_dir: str,
    chunk_mode: str,
    slides_per_chunk: int,
    minutes_per_chunk: float,
    questions_per_chunk: int,
    model: str,
    rag_k: int,
    output_path: str | None,
    embedding_model: str,
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

    print(f"[INFO] チャンク数: {len(chunks)}  モード: {mode_label}  1区間{questions_per_chunk}問  モデル: {model}")
    print(f"[INFO] RAGインデックス: {chroma_dir}")

    child_store, parent_store = load_stores(chroma_dir, embedding_model)

    results = []
    for chunk in chunks:
        label = f"{chunk['chunk_id']} ({chunk['start_sec']:.0f}s〜{chunk['end_sec']:.0f}s)"
        print(f"  {label}", end=" ", flush=True)

        # Step 1: seed抽出
        seeds = extract_seeds(chunk["ocr_text"], chunk["asr_text"], model, lmstudio_url)
        if not seeds:
            print("SKIP (seed抽出失敗)")
            continue
        print(f"seed:{len(seeds)}語", end=" ", flush=True)

        # Step 2: RAG検索（全seedを1クエリにまとめて検索）
        query = "、".join(seeds)
        docs = search(query, child_store, parent_store, k=rag_k)
        if not docs:
            print("SKIP (RAGヒットなし)")
            continue
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Step 3: 質問生成
        questions = generate_questions_with_rag(seeds, context, questions_per_chunk, model, lmstudio_url)
        if questions:
            results.append({
                "chunk_id":  chunk["chunk_id"],
                "slide_ids": chunk["slide_ids"],
                "start_sec": chunk["start_sec"],
                "end_sec":   chunk["end_sec"],
                "seeds":     seeds,
                "questions": questions,
            })
            print(f"{len(questions)}問 OK")
        else:
            print("SKIP (質問生成失敗)")

    out_path = (
        Path(output_path) if output_path
        else Path(result_json_path).parent / f"qg_rag_{chunk_mode}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = sum(len(r["questions"]) for r in results)
    print(f"\n[DONE] {total}問生成（{len(results)}チャンク） → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RAGを使って質問生成（パターン3・4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result",    help="result.jsonのパス")
    parser.add_argument("--chroma",  required=True, help="RAGインデックスのパス")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--by-slides", action="store_true", help="スライド枚数基準（パターン3）")
    mode_group.add_argument("--by-time",   action="store_true", help="経過時間基準（パターン4）")

    parser.add_argument("--slides-per-chunk",    type=int,   default=3,   help="スライド枚数基準のチャンクサイズ")
    parser.add_argument("--minutes-per-chunk",   type=float, default=5.0, help="経過時間基準のチャンクサイズ（分）")
    parser.add_argument("--questions-per-chunk", type=int,   default=1,   help="1区間あたりの生成問題数")
    parser.add_argument("--model",           default="qwen/qwen3-vl-8b",       help="LMStudioモデル名")
    parser.add_argument("--rag-k",           type=int, default=3,              help="RAG検索の上位k件")
    parser.add_argument("--output",          default=None,                      help="出力JSONパス（省略時: result.jsonと同ディレクトリ）")
    parser.add_argument("--embedding-model", default="cl-nagoya/ruri-v3-310m", help="埋め込みモデル名")
    parser.add_argument("--lmstudio-url",    default=LMSTUDIO_URL,              help="LMStudioエンドポイント")

    args = parser.parse_args()
    chunk_mode = "slides" if args.by_slides else "time"

    run(
        result_json_path=args.result,
        chroma_dir=args.chroma,
        chunk_mode=chunk_mode,
        slides_per_chunk=args.slides_per_chunk,
        minutes_per_chunk=args.minutes_per_chunk,
        questions_per_chunk=args.questions_per_chunk,
        model=args.model,
        rag_k=args.rag_k,
        output_path=args.output,
        embedding_model=args.embedding_model,
        lmstudio_url=args.lmstudio_url,
    )


if __name__ == "__main__":
    main()
