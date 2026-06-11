"""
eval_rag_quality.py

RAGコンテキストを付与したLLM評価と人手評価のスペアマン相関を検証する。

RAG構成（6パターン）× LLM（2種）× 動画（2本）= 24条件
  char_asr+ocr  : chroma_char_asr 上位k件 + chroma_char_ocr 上位k件 → LLMに2k件渡す
  char_mix      : chroma_char_asr_ocr_mix 上位k件
  char_aligned  : chroma_char_asr で上位k/2件検索→OCR逆引き + chroma_char_ocr で上位k/2件検索→ASR逆引き
  slide_asr+ocr : chroma_slide_asr 上位k件 + chroma_slide_ocr 上位k件 → LLMに2k件渡す
  slide_mix     : chroma_slide_asr_ocr_mix 上位k件
  slide_aligned : chroma_slide_asr で上位k/2件検索→OCR逆引き + chroma_slide_ocr で上位k/2件検索→ASR逆引き

使い方:
  python eval_rag_quality.py
  python eval_rag_quality.py --models gpt-oss-20b qwen/qwen3-vl-30b
  python eval_rag_quality.py --models gpt-oss-20b --n-trials 3 --top-k 3
  python eval_rag_quality.py --video 31   # video31のみ実行

出力:
  previous_qg/rag_quality_scores.json  （途中経過も逐次保存）
  previous_qg/rag_quality_result.json  （完了時サマリ）
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from scipy.stats import spearmanr
from langchain_chroma import Chroma

from rag.retriever import search_aligned

# ─── パス設定 ──────────────────────────────────────
BASE          = Path(__file__).parent
ROOT          = BASE.parent
HUMAN_EVAL    = BASE / "previous_qg/human_eval.json"
PREV_AUTO_EVAL = BASE / "previous_qg/eval-result.json"
SCORES_FILE   = BASE / "previous_qg/rag_quality_scores.json"
RESULT_FILE   = BASE / "previous_qg/rag_quality_result.json"

LMSTUDIO_URL  = "http://localhost:1234/v1/chat/completions"

# video_id → ChromaDB ベースディレクトリ のマッピング
VIDEO_CHROMA_BASE = {
    31: ROOT / "movie/test1",
    32: ROOT / "movie/test2",
}

# RAG構成: dirs[0]=ASR, dirs[1]=OCR の順で指定する
# aligned=True の場合は search_aligned で相互逆引きし、dirs は [asr_dir, ocr_dir] の順
RAG_CONFIGS = [
    {"name": "char_asr+ocr",  "dirs": ["chroma_char_asr",          "chroma_char_ocr"]},
    {"name": "char_mix",      "dirs": ["chroma_char_asr_ocr_mix"]},
    {"name": "char_aligned",  "dirs": ["chroma_char_asr",          "chroma_char_ocr"],  "aligned": True},
    {"name": "slide_asr+ocr", "dirs": ["chroma_slide_asr",         "chroma_slide_ocr"]},
    {"name": "slide_mix",     "dirs": ["chroma_slide_asr_ocr_mix"]},
    {"name": "slide_aligned", "dirs": ["chroma_slide_asr",         "chroma_slide_ocr"], "aligned": True},
]

EVAL_PROMPT = """\
あなたは大学講義の問題品質評価者です。
以下の講義資料を参照して、問題と解答の正確性を評価してください。

# 講義資料（RAGで取得）
{rag_context}

# 評価対象
問題文: {question}
解答文: {answer}

# タスク内容
「この問題は正確性という観点で優れている」ということについて、
あなたがどう思うか、以下の尺度で評価を行ってください。
- 1.全くそう思わない
- 2.あまりそう思わない
- 3.どちらとも言えない
- 4.ややそう思う
- 5.とてもそう思う


## 評価軸
正確性

## 正確性の説明
正確であるとは、問題と解答が講義音声や講義スライドに正確に基づいているということです。これは、表現と内容について以下を意味します。
 - 表現：問題と解答に、誤字脱字、曖昧で誤解を招く表現がない
 - 内容：問題と解答が、講義音声や講義スライドに基づいており、問題と解答の内容に整合性がある

## 注意点
評価の思考過程をcot（Chain of Thought）で詳しく説明してください。スコアの理由も具体的に述べてください。

# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が {{ で最後の文字が }} であること。

{{
  "cot": "思考過程をここに記述",
  "score": 0,
  "reason": "スコアの理由"
}}"""


# ─── ChromaDB ロード ────────────────────────────────
def load_stores(chroma_dir: Path, embeddings) -> tuple[Chroma, Chroma]:
    child_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    parent_store = Chroma(
        collection_name="parent_chunks",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    return child_store, parent_store


SCORE_THRESHOLD = 0.40  # squared L2 距離の上限（コサイン類似度 0.80 以上のみ採用）


# ─── 1つの DB から上位 top_k 件の親テキストを取得 ──────
def retrieve_from_store(query: str, child_store: Chroma, parent_store: Chroma,
                        top_k: int) -> list[str]:
    hits_with_score = child_store.similarity_search_with_score(query, k=top_k * 2)

    # スコアフィルタ（距離 < SCORE_THRESHOLD のみ採用、なければ上位1件）
    filtered = [(doc, s) for doc, s in hits_with_score if s < SCORE_THRESHOLD]
    if not filtered:
        filtered = hits_with_score[:1]

    seen, chunk_ids = set(), []
    for doc, _ in filtered:
        cid = doc.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            chunk_ids.append(cid)
            if len(chunk_ids) >= top_k:
                break

    texts = []
    for cid in chunk_ids:
        res = parent_store.get(ids=[cid], include=["documents"])
        if res["documents"]:
            texts.append(res["documents"][0])
    return texts


# ─── RAG コンテキスト取得（複数 DB 対応） ───────────────
def retrieve_context(query: str,
                     store_pairs: list[tuple[Chroma, Chroma]],
                     top_k: int = 3) -> str:
    """
    store_pairs: [(child_store, parent_store), ...] のリスト
    各 DB から top_k 件ずつ取得して結合する。
    """
    all_texts = []
    for child_store, parent_store in store_pairs:
        texts = retrieve_from_store(query, child_store, parent_store, top_k)
        all_texts.extend(texts)
    parts = [f"検索結果{i+1}：\n{text}" for i, text in enumerate(all_texts)]
    return "\n\n".join(parts) if parts else ""


# ─── aligned 検索（ASR/OCR 相互逆引き） ─────────────
def retrieve_aligned_context(query: str,
                              asr_child: Chroma, asr_parent: Chroma,
                              ocr_child: Chroma, ocr_parent: Chroma,
                              top_k: int) -> str:
    docs = search_aligned(query, asr_child, asr_parent, ocr_child, ocr_parent, k=top_k)
    parts = [f"検索結果{i+1}：\n{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n".join(parts) if parts else ""


# ─── LLM 評価（1回） ────────────────────────────────
def call_llm(question: str, answer: str, context: str,
             model: str, url: str,
             temperature: float = 1.0,
             max_retries: int = 2) -> int | None:
    prompt = EVAL_PROMPT.format(
        rag_context=context or "（取得できませんでした）",
        question=question,
        answer=answer,
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": temperature,
    }
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # JSON 抽出（コードブロックを除去）
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            # 最初の { から最後の } までを取り出す
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                score = data.get("score")
                if isinstance(score, (int, float)) and 1 <= score <= 5:
                    return int(round(score))
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"      [WARN] LLM呼び出し失敗: {e}", flush=True)
    return None


# ─── 5回評価して平均 ─────────────────────────────────
def evaluate_question(question: str, answer: str, context: str,
                      model: str, url: str, n_trials: int,
                      temperature: float) -> tuple[list[int], float | None]:
    scores = []
    for _ in range(n_trials):
        s = call_llm(question, answer, context, model, url, temperature)
        if s is not None:
            scores.append(s)
    avg = sum(scores) / len(scores) if scores else None
    return scores, avg


# ─── チェックポイント保存/ロード ──────────────────────
def load_scores(settings: dict) -> dict:
    """
    設定が変わった場合（n_trials 等）はチェックポイントを無効化してリセットする。
    """
    if SCORES_FILE.exists():
        with open(SCORES_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        saved_settings = saved.get("_settings", {})
        # n_trials・top_k・temperature が一致しない場合はリセット
        mismatch = any(saved_settings.get(k) != v
                       for k, v in settings.items())
        if mismatch:
            print(f"[checkpoint] 設定変更を検出 → チェックポイントをリセット")
            print(f"  保存済み: {saved_settings}")
            print(f"  現在値:   {settings}")
            return {"_settings": settings}
        return saved
    return {"_settings": settings}


def save_scores(data: dict):
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── 相関計算 ──────────────────────────────────────
def spearman(x: list, y: list) -> tuple[float, float] | tuple[None, None]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None, None
    xs, ys = zip(*pairs)
    r, p = spearmanr(xs, ys)
    return float(r), float(p)


# ─── ベースライン（RAGなし自動評価）の相関 ─────────────
def compute_baseline(human_qs: list[dict], auto_data: dict,
                     video_id: int, models: list[str]) -> dict:
    auto_map = {q["qg_id"]: q for q in auto_data["questions"]}
    results = {}
    for model in models:
        teacher_scores, auto_scores = [], []
        for q in human_qs:
            if q["video_id"] != video_id:
                continue
            aq = auto_map.get(q["qg_id"])
            if aq is None:
                continue
            by_model = aq.get("accuracy_by_model", {})
            if model not in by_model:
                continue
            teacher_scores.append(q["accuracy_avg"])
            auto_scores.append(by_model[model])
        r, p = spearman(teacher_scores, auto_scores)
        results[model] = {"spearman_r": r, "p_value": p, "n": len(teacher_scores)}
    return results


# ─── メイン ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAGコンテキスト付きLLM評価 vs 人手評価 スペアマン相関検証",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+",
                        default=["gpt-oss-20b", "qwen/qwen3-vl-30b"],
                        help="LMStudio モデル名（スペース区切りで複数指定可）")
    parser.add_argument("--n-trials",    type=int,   default=5)
    parser.add_argument("--top-k",       type=int,   default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--url",         default=LMSTUDIO_URL)
    parser.add_argument("--video",       type=int,   choices=[31, 32],
                        help="特定 video_id のみ実行（省略時: 利用可能な全動画）")
    args = parser.parse_args()

    # ─── データ読み込み ──────────────────────────────
    with open(HUMAN_EVAL, encoding="utf-8") as f:
        human_data = json.load(f)
    with open(PREV_AUTO_EVAL, encoding="utf-8") as f:
        auto_data = json.load(f)

    human_qs = human_data["questions"]
    print(f"[data] 人手評価問数: {len(human_qs)}")

    # ─── 利用可能なインデックスを確認 ────────────────
    available_videos = []
    for vid, base in VIDEO_CHROMA_BASE.items():
        if args.video and args.video != vid:
            continue
        needed = {d for cfg in RAG_CONFIGS for d in cfg["dirs"]}
        if any((base / d).exists() for d in needed):
            available_videos.append(vid)
        else:
            print(f"[SKIP] video{vid}: {base} にインデックスが見つかりません")

    if not available_videos:
        print("[ERROR] 利用可能なインデックスがありません。終了します。")
        sys.exit(1)

    # ─── 埋め込みモデルをロード（1回だけ） ────────────
    from rag.embeddings import RuriEmbeddings
    print("\n[embeddings] モデルロード中...")
    embeddings = RuriEmbeddings("cl-nagoya/ruri-v3-310m")

    # ─── チェックポイントのロード ─────────────────────
    ckpt_settings = {
        "n_trials":    args.n_trials,
        "top_k":       args.top_k,
        "temperature": args.temperature,
    }
    scores_data = load_scores(ckpt_settings)

    # ─── 実行ループ ──────────────────────────────────
    print(f"\n設定: models={args.models}  n_trials={args.n_trials}"
          f"  top_k={args.top_k}  temperature={args.temperature}\n")

    for video_id in available_videos:
        chroma_base = VIDEO_CHROMA_BASE[video_id]
        target_qs = [q for q in human_qs if q["video_id"] == video_id]
        print(f"\n{'━'*60}")
        print(f"  video_id={video_id}  ({len(target_qs)}問)")
        print(f"{'━'*60}")

        for cfg in RAG_CONFIGS:
            rag_name   = cfg["name"]
            is_aligned = cfg.get("aligned", False)

            # 必要な DB が存在するか確認
            missing = [d for d in cfg["dirs"] if not (chroma_base / d).exists()]
            if missing:
                print(f"  [SKIP] {rag_name}: {missing} が見つかりません")
                continue

            print(f"\n  ▶ {rag_name}  ({' + '.join(cfg['dirs'])})")

            # DB ロード
            if is_aligned:
                # dirs = [asr_dir, ocr_dir]
                asr_child, asr_parent = load_stores(chroma_base / cfg["dirs"][0], embeddings)
                ocr_child, ocr_parent = load_stores(chroma_base / cfg["dirs"][1], embeddings)
                store_pairs = None
            else:
                store_pairs = [
                    load_stores(chroma_base / d, embeddings)
                    for d in cfg["dirs"]
                ]

            for model in args.models:
                cond_key = f"v{video_id}_{rag_name}_{model}"
                if cond_key not in scores_data:
                    scores_data[cond_key] = {}

                print(f"    [{model}]", flush=True)
                completed = len(scores_data[cond_key])
                print(f"      チェックポイント: {completed}/{len(target_qs)} 問完了済み")

                for q in target_qs:
                    qid = str(q["qg_id"])
                    if qid in scores_data[cond_key]:
                        continue  # 再開: 完了済みはスキップ

                    query = q["question"] + " " + q["answer"]
                    if is_aligned:
                        context = retrieve_aligned_context(
                            query, asr_child, asr_parent, ocr_child, ocr_parent, args.top_k
                        )
                    else:
                        context = retrieve_context(query, store_pairs, args.top_k)
                    trial_scores, avg = evaluate_question(
                        q["question"], q["answer"], context,
                        model, args.url, args.n_trials, args.temperature,
                    )
                    scores_data[cond_key][qid] = {
                        "trial_scores": trial_scores,
                        "avg":          avg,
                        "teacher":      q["accuracy_avg"],
                        "rag_context":  context,
                    }
                    save_scores(scores_data)  # 逐次保存

                    done = sum(1 for v in scores_data[cond_key].values()
                               if v["avg"] is not None)
                    total = len(target_qs)
                    print(f"      [{done:3d}/{total}] qg_id={qid}"
                          f"  trials={trial_scores}  avg={avg}  "
                          f"teacher={q['accuracy_avg']}",
                          flush=True)

                # 相関計算
                t_scores, r_scores = [], []
                for v in scores_data[cond_key].values():
                    if v["avg"] is not None:
                        t_scores.append(v["teacher"])
                        r_scores.append(v["avg"])
                corr, pval = spearman(t_scores, r_scores)
                sig = "**" if pval is not None and pval < 0.01 else (
                      "*"  if pval is not None and pval < 0.05 else "  ")
                print(f"      → スペアマン r={corr:+.4f}  p={pval:.4f} {sig}  "
                      f"(n={len(t_scores)})")

    # ─── 最終サマリ ─────────────────────────────────
    print(f"\n\n{'═'*65}")
    print("  最終結果サマリ")
    print(f"{'═'*65}")

    summary_rows = []
    for video_id in available_videos:
        target_qs = [q for q in human_qs if q["video_id"] == video_id]

        # ベースライン（RAGなし）
        baseline = compute_baseline(human_qs, auto_data, video_id, args.models)
        for model, b in baseline.items():
            r, p = b["spearman_r"], b["p_value"]
            if r is not None:
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
                print(f"  [baseline] v{video_id}  {model:<28}  r={r:+.4f}  "
                      f"p={p:.4f} {sig}")
                summary_rows.append({
                    "video_id": video_id, "rag_type": "baseline(no-rag)",
                    "model": model, "spearman_r": round(r, 6),
                    "p_value": round(p, 6), "n": b["n"],
                })

        print()
        for cfg in RAG_CONFIGS:
            for model in args.models:
                cond_key = f"v{video_id}_{cfg['name']}_{model}"
                if cond_key not in scores_data:
                    continue
                t_scores, r_scores = [], []
                for v in scores_data[cond_key].values():
                    if v["avg"] is not None:
                        t_scores.append(v["teacher"])
                        r_scores.append(v["avg"])
                corr, pval = spearman(t_scores, r_scores)
                if corr is None:
                    continue
                sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "  ")
                rag_name = cfg["name"]
                print(f"  v{video_id}  {rag_name:<16}  [{model:<22}]  "
                      f"r={corr:+.4f}  p={pval:.4f} {sig}")
                summary_rows.append({
                    "video_id": video_id, "rag_type": rag_name,
                    "model": model, "spearman_r": round(corr, 6),
                    "p_value": round(pval, 6), "n": len(t_scores),
                })
        print()

    print(f"{'═'*65}")

    # ─── 結果 JSON 保存 ──────────────────────────────
    result = {
        "settings": {
            "models":      args.models,
            "n_trials":    args.n_trials,
            "top_k":       args.top_k,
            "temperature": args.temperature,
        },
        "summary": sorted(summary_rows,
                          key=lambda x: (x["video_id"], x["spearman_r"]),
                          reverse=True),
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n詳細スコア: {SCORES_FILE}")
    print(f"サマリ:     {RESULT_FILE}")


if __name__ == "__main__":
    main()
