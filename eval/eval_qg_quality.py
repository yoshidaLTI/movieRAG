"""
eval_qg_quality.py

qg_result.json の問題（時間ベース3手法）をLLMで評価し、メソッド別に集計する。

評価対象:
  - direct  (chunk_mode=time) … パターン2
  - rag     (chunk_mode=time) … パターン4
  - rag_cot (chunk_mode=time) … パターン6

評価軸は外部から指定可能:
  --axes 正確性 明確性          # 組み込み軸を選択
  --axes-file my_axes.json      # カスタム定義 {"軸名": "説明", ...}

使い方:
  python eval/eval_qg_quality.py movie/大分大学入門/qg_result.json

  # 評価軸を絞る
  python eval/eval_qg_quality.py movie/大分大学入門/qg_result.json \\
      --axes 正確性 識別性

  # カスタム軸
  python eval/eval_qg_quality.py movie/大分大学入門/qg_result.json \\
      --axes-file eval/my_axes.json

  # モデル・試行数を変更
  python eval/eval_qg_quality.py movie/大分大学入門/qg_result.json \\
      --models gpt-oss-20b --n-trials 5

出力:
  {video_dir}/qg_eval_scores.json   詳細スコア（チェックポイント兼用）
  {video_dir}/qg_eval_result.json   メソッド別サマリ
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import load_stores, search

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

TARGET_METHODS = {"direct", "rag", "rag_cot"}

# ─── 組み込み評価軸 ─────────────────────────────────────────
BUILTIN_AXES: dict[str, str] = {
    "正確性": (
        "正確であるとは、問題と解答が講義音声や講義スライドに正確に基づいているということです。これは、表現と内容について以下を意味します。"
        " - 表現：問題と解答に、誤字脱字、曖昧で誤解を招く表現がない"
        " - 内容：問題と解答が、講義音声や講義スライドに基づいており、問題と解答の内容に整合性がある"
    ),
    "解答可能性": (
        "解答可能であるとは、講義を受講した学生であれば解答できる問題であるということです。これは、問題文と参照ドキュメントについて以下を意味します。"
        " - 問題文：問題文が明確で、解答するために必要な情報が提供されている"
        " - 講義資料：講義資料を参照すれば、正解を導き出せる状態である"
    ),
    "Bloomレベルの適切さ": (
        "Bloomの教育目標分類（知識・応用・評価）に沿った難易度・深さで出題されているかを評価する。"
        " - 知識: 講義内容の事実や定義を問う問題"
        " - 応用: 講義内容を理解し、具体的な状況に適用する能力を問う問題"
        " - 評価: 講義内容を分析・評価し、独自の見解を形成する能力を問う問題"
    ),
    "誤答の品質": (
        "誤答の品質が高いとは、誤答が講義内容に関連しつつも、正解と区別できるような内容であるということです。これは、誤答の内容と難易度について以下を意味します。"
        " - 内容：誤答が講義内容に関連しているが、正解とは異なる内容である"
        " - 難易度：誤答があまりにも明らかではなく、適切な難易度である"
    ),
}

# ─── プロンプト（1評価軸ずつ呼び出す） ──────────────────────
EVAL_PROMPT = """\
以下の講義資料を参照して、4択問題の品質を評価してください。

# 講義資料（RAGで取得）
{rag_context}

# 評価対象
問題文: {question}
選択肢A: {choice_A}
選択肢B: {choice_B}
選択肢C: {choice_C}
選択肢D: {choice_D}
正解: {answer}
正解の理由: {reason}

# タスク内容
「この問題は{axis_name}という観点で優れている」ということについて、
あなたがどう思うか、以下の尺度で評価を行ってください。
- 1.全くそう思わない
- 2.あまりそう思わない
- 3.どちらとも言えない
- 4.ややそう思う
- 5.とてもそう思う

## 評価軸
{axis_name}

## {axis_name}の説明
{axis_desc}

## 注意点
評価の思考過程をcot（Chain of Thought）で詳しく説明してください。スコアの理由も具体的に述べてください。

# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が {{ で最後の文字が }} であること。

{{"cot": "思考過程をここに記述", "score": 0, "reason": "スコアの理由"}}"""


def _build_prompt(q: dict, rag_context: str, axis_name: str, axis_desc: str) -> str:
    return EVAL_PROMPT.format(
        rag_context=rag_context or "（取得できませんでした）",
        question=q["question"],
        choice_A=q.get("choice_A", ""),
        choice_B=q.get("choice_B", ""),
        choice_C=q.get("choice_C", ""),
        choice_D=q.get("choice_D", ""),
        answer=q.get("answer", ""),
        reason=q.get("reason", ""),
        axis_name=axis_name,
        axis_desc=axis_desc,
    )


# ─── LLM 呼び出し（1軸・1回） ────────────────────────────────
def _call_llm(
    prompt: str,
    model: str,
    url: str,
    temperature: float = 1.0,
    max_retries: int = 2,
) -> dict | None:
    """score / cot / reason を含む dict を返す。スコア取得失敗時は None。"""
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
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                v = data.get("score")
                if isinstance(v, (int, float)) and 1 <= v <= 5:
                    return {
                        "score":  int(round(v)),
                        "cot":    str(data.get("cot", "")),
                        "reason": str(data.get("reason", "")),
                    }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"      [WARN] LLM呼び出し失敗: {e}", flush=True)
    return None


# ─── 1問を全軸 × n_trials 回評価 ─────────────────────────────
def _evaluate(
    q: dict,
    rag_context: str,
    axes: dict[str, str],
    model: str,
    url: str,
    n_trials: int,
    temperature: float,
) -> dict[str, list[dict]]:
    collected: dict[str, list[dict]] = {name: [] for name in axes}
    for axis_name, axis_desc in axes.items():
        prompt = _build_prompt(q, rag_context, axis_name, axis_desc)
        for _ in range(n_trials):
            result = _call_llm(prompt, model, url, temperature)
            if result is not None:
                collected[axis_name].append(result)
    return collected


# ─── 評価軸ロード ─────────────────────────────────────────────
def _load_axes(args: argparse.Namespace) -> dict[str, str]:
    if args.axes_file:
        with open(args.axes_file, encoding="utf-8") as f:
            return json.load(f)
    if args.axes:
        selected = {}
        for name in args.axes:
            if name in BUILTIN_AXES:
                selected[name] = BUILTIN_AXES[name]
            else:
                print(f"[WARN] 未知の評価軸: {name}  (利用可能: {list(BUILTIN_AXES)})")
        return selected or BUILTIN_AXES
    return BUILTIN_AXES


# ─── チェックポイント ─────────────────────────────────────────
def _load_checkpoint(path: Path, settings: dict) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("_settings") != settings:
            print(f"[checkpoint] 設定変更を検出 → リセット")
            print(f"  保存済み: {saved.get('_settings')}")
            print(f"  現在値  : {settings}")
            return {"_settings": settings}
        return saved
    return {"_settings": settings}


def _save_checkpoint(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── メイン ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="qg_result.json の時間ベース問題をLLM評価してメソッド別に集計する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("qg_result", help="qg_result.jsonのパス")
    parser.add_argument(
        "--chroma", default=None,
        help="RAGインデックスのパス（省略時: qg_result.jsonと同ディレクトリのchroma_char_asr_ocr_mix）",
    )
    parser.add_argument(
        "--axes", nargs="+", default=None, metavar="軸名",
        help=f"使用する組み込み評価軸（省略時: 全軸）。選択肢: {list(BUILTIN_AXES)}",
    )
    parser.add_argument(
        "--axes-file", default=None,
        help='カスタム評価軸の定義JSONファイル（{"軸名": "説明", ...}形式）',
    )
    parser.add_argument("--models", nargs="+", default=["gpt-oss-20b"], help="LLM評価モデル")
    parser.add_argument("--n-trials",    type=int,   default=1,   help="1問あたりのLLM評価回数")
    parser.add_argument("--top-k",       type=int,   default=2,   help="RAG検索の上位k件")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--embedding-model", default="cl-nagoya/ruri-v3-310m")
    parser.add_argument(
        "--output", default=None,
        help="出力先ディレクトリ（省略時: qg_result.jsonと同ディレクトリ）",
    )
    parser.add_argument("--url", default=LMSTUDIO_URL, help="LMStudioエンドポイント")
    parser.add_argument(
        "--save-cot", action="store_true",
        help="評価理由（cot・reason）をqg_eval_scores.jsonに保存する（省略時: スコアのみ保存）",
    )
    args = parser.parse_args()

    qg_path = Path(args.qg_result)
    out_dir = Path(args.output) if args.output else qg_path.parent
    scores_path = out_dir / "qg_eval_scores.json"
    result_path = out_dir / "qg_eval_result.json"

    if args.chroma is None:
        args.chroma = str(qg_path.parent / "chroma_char_asr_ocr_mix")

    axes = _load_axes(args)
    print(f"[INFO] 評価軸: {list(axes)}")

    with open(qg_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    target_qs = [
        q for q in all_questions
        if q.get("method") in TARGET_METHODS
        and q.get("detail_setting", {}).get("chunk_mode") == "time"
    ]
    print(f"[INFO] 評価対象: {len(target_qs)}問 / 全{len(all_questions)}問")
    method_counts = {}
    for q in target_qs:
        method_counts[q["method"]] = method_counts.get(q["method"], 0) + 1
    for m, c in method_counts.items():
        print(f"  {m}: {c}問")

    if not target_qs:
        print("[ERROR] 評価対象の問題がありません（chunk_mode=time の3手法が必要です）")
        sys.exit(1)

    # RAG ロード
    print(f"\n[INFO] 埋め込みモデルをロード中... ({args.embedding_model})")
    child_store, parent_store = load_stores(args.chroma, args.embedding_model)
    print(f"[INFO] RAGインデックス: {args.chroma}")

    # チェックポイント
    ckpt_settings = {
        "n_trials":    args.n_trials,
        "top_k":       args.top_k,
        "temperature": args.temperature,
        "axes":        list(axes),
    }
    scores_data = _load_checkpoint(scores_path, ckpt_settings)

    print(f"\n設定: models={args.models}  n_trials={args.n_trials}"
          f"  top_k={args.top_k}  temperature={args.temperature}\n")

    for model in args.models:
        print(f"\n{'━'*60}")
        print(f"  モデル: {model}  ({len(target_qs)}問)")
        print(f"{'━'*60}")

        completed = sum(
            1 for k in scores_data
            if isinstance(scores_data[k], dict) and k.endswith(f"__{model}")
            and set(scores_data[k].get("axes", {}).keys()) >= set(axes)
        )
        print(f"  チェックポイント: {completed}/{len(target_qs)} 問完了済み\n")

        for qi, q in enumerate(target_qs):
            qid = q["question_id"]
            ckpt_key = f"{qid}__{model}"

            # 全軸が既に揃っていればスキップ
            if ckpt_key in scores_data:
                existing = set(scores_data[ckpt_key].get("axes", {}).keys())
                if existing >= set(axes):
                    continue

            method = q["method"]
            mode   = q.get("detail_setting", {}).get("chunk_mode", "?")
            label  = f"[{qi+1:3d}/{len(target_qs)}] {method}/{mode}  {qid}"
            print(f"  {label}", end=" ", flush=True)

            # RAG 検索
            docs = search(q["question"], child_store, parent_store, k=args.top_k)
            rag_context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""

            # LLM 評価
            trial_results = _evaluate(q, rag_context, axes, model, args.url,
                                       args.n_trials, args.temperature)

            axis_summary: dict[str, dict] = {}
            for name, trial_results_per_axis in trial_results.items():
                scores = [r["score"] for r in trial_results_per_axis]
                avg = sum(scores) / len(scores) if scores else None
                entry: dict = {"trial_scores": scores, "avg": avg}
                if args.save_cot:
                    entry["cot_list"]    = [r["cot"]    for r in trial_results_per_axis]
                    entry["reason_list"] = [r["reason"] for r in trial_results_per_axis]
                axis_summary[name] = entry

            scores_data[ckpt_key] = {
                "method":      method,
                "chunk_mode":  mode,
                "question":    q.get("question", ""),
                "choice_A":    q.get("choice_A", ""),
                "choice_B":    q.get("choice_B", ""),
                "choice_C":    q.get("choice_C", ""),
                "choice_D":    q.get("choice_D", ""),
                "answer":      q.get("answer", ""),
                "reason":      q.get("reason", ""),
                "axes":        axis_summary,
                "rag_context": rag_context,
            }
            _save_checkpoint(scores_data, scores_path)

            score_str = "  ".join(
                f"{name}={v['avg']:.1f}" if v["avg"] is not None else f"{name}=?"
                for name, v in axis_summary.items()
            )
            print(score_str)

    # ─── サマリ集計 ──────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print("  メソッド別スコアサマリ")
    print(f"{'═'*65}")

    method_labels = [
        ("direct",  "パターン2: 直接生成（時間）"),
        ("rag",     "パターン4: RAG（時間）"),
        ("rag_cot", "パターン6: RAG+CoT（時間）"),
    ]

    summary_rows: list[dict] = []
    for model in args.models:
        print(f"\n  [{model}]")
        for method, label in method_labels:
            axis_scores: dict[str, list[float]] = {name: [] for name in axes}
            for ckpt_key, v in scores_data.items():
                if not isinstance(v, dict) or not ckpt_key.endswith(f"__{model}"):
                    continue
                if v.get("method") != method or v.get("chunk_mode") != "time":
                    continue
                for name in axes:
                    avg = v.get("axes", {}).get(name, {}).get("avg")
                    if avg is not None:
                        axis_scores[name].append(avg)

            n = max((len(s) for s in axis_scores.values()), default=0)
            if n == 0:
                continue

            print(f"\n    {label}  (n={n})")
            row: dict = {"model": model, "method": method, "chunk_mode": "time", "n": n, "axes": {}}
            for name in axes:
                scores_list = axis_scores[name]
                mean = sum(scores_list) / len(scores_list) if scores_list else None
                mean_str = f"{mean:.2f}" if mean is not None else "  -  "
                print(f"      {name:8s}: {mean_str}")
                row["axes"][name] = round(mean, 4) if mean is not None else None
            summary_rows.append(row)

    print(f"\n{'═'*65}")

    result = {
        "settings": {
            "models":      args.models,
            "n_trials":    args.n_trials,
            "top_k":       args.top_k,
            "temperature": args.temperature,
            "axes":        {name: desc for name, desc in axes.items()},
        },
        "summary": summary_rows,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n詳細スコア: {scores_path}")
    print(f"サマリ:     {result_path}")


if __name__ == "__main__":
    main()
