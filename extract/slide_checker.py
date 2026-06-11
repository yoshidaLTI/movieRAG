"""
slide_checker.py

動画からスライドの変化・アニメーションを検知し、
OCR結果をJSON＋画像ディレクトリで出力するツール。

検知ロジック:
  1. ffmpeg scdet フィルタで scene_score を取得（0〜100スケール）
     threshold=0.001 で変化フレームを広めに取得
  2. SCENE_FINE(0.005) 未満のフレームは処理対象外として足切り
  3. 検知フレームを SSIM ブロック分割で前後比較
     SCENE_COARSE(5.0) と SSIM分布を組み合わせて change/animation を確定
  4. 前回OCRから OCR_INTERVAL 秒以内はスキップ
  5. LMStudio(GLM-OCR) で OCR
  6. 出力:
       <video_stem>/slide1/picture1.jpg ...
       <video_stem>/result.json
"""

import os
import re
import json
import base64
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import requests

# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────
# ffmpeg scdet スコアは 0〜100 スケール
# 実測値参考: 30.694=明確な切り替え, 0.007〜3.969=アニメーション/小変化
SCENE_COARSE = 5.0           # これ以上：明確な変化として SSIM 分布で change/animation を判定
SCENE_FINE   = 0.005         # これ未満：処理対象外として足切り

SSIM_GRID             = 4    # ブロック分割数（4×4=16ブロック）
SSIM_CHANGE_THRESHOLD = 0.6  # 変化ブロック比率がこれ以上 → change寄り
SSIM_SAME_THRESHOLD   = 0.99 # SSIMがこれ以上 → 前フレームと同一とみなしスキップ

OCR_INTERVAL = 1.5           # 前回OCRから何秒以内はスキップするか
OCR_DELAY    = 1.0           # 検知から何秒待ってOCRするか（アニメーション完了待ち）

LMSTUDIO_URL   = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "glm-ocr"  # LMStudioで定義したOCR特化モデル名

OCR_PROMPT = (
    "この画像に写っているテキストをすべて正確に読み取り、"
    "そのままの形式で出力してください。"
    "余計な説明は不要です。テキストのみ出力してください。"
)


# ────────────────────────────────────────────
# ffmpeg scdet: scene_score 取得
# ────────────────────────────────────────────
def get_scene_scores_via_filter(video_path: str) -> list[dict]:
    """
    ffmpeg の scdet フィルタで scene_score を取得する。

    ffmpeg 8.x の出力形式:
      [Parsed_scdet_0 @ ...] lavfi.scd.score: 30.694, lavfi.scd.time: 47.652561

    スコアは 0〜100 スケール。
    threshold=0.001 で変化フレームを広めに取得し、
    Python側で SCENE_FINE(0.005) 未満を足切りする。
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "scdet=threshold=0.001:sc_pass=1",
        "-an",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stderr
    except Exception as e:
        print(f"[ERROR] ffmpeg scdet 失敗: {e}")
        return []

    frames = []
    for line in output.splitlines():
        if "lavfi.scd.score" not in line:
            continue
        score_match = re.search(r"lavfi\.scd\.score:\s*([\d.]+)", line)
        time_match  = re.search(r"lavfi\.scd\.time:\s*([\d.]+)", line)
        if score_match and time_match:
            frames.append({
                "time":  float(time_match.group(1)),
                "score": float(score_match.group(1)),
            })

    print(f"[INFO] scdet 検知フレーム数（raw）: {len(frames)}")
    return frames


# ────────────────────────────────────────────
# フレーム抽出
# ────────────────────────────────────────────
def extract_frame(video_path: str, timestamp: float) -> np.ndarray | None:
    """指定タイムスタンプのフレームを numpy 配列で返す。"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ────────────────────────────────────────────
# SSIM ブロック分割
# ────────────────────────────────────────────
def ssim_block_analysis(frame1: np.ndarray, frame2: np.ndarray, grid: int = SSIM_GRID) -> dict:
    """
    2フレームを grid×grid に分割し、各ブロックの SSIM を計算する。

    戻り値:
        {
            "block_scores":   [[float, ...], ...],  # grid×grid の SSIM 値
            "changed_ratio":  float,                # 変化ブロックの比率
            "is_distributed": bool,                 # 変化が全体的かどうか
            "mean_ssim":      float,                # 全体平均 SSIM
        }
    """
    h, w = frame1.shape[:2]
    bh, bw = h // grid, w // grid

    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    block_scores   = []
    changed_blocks = 0

    for row in range(grid):
        row_scores = []
        for col in range(grid):
            y1, y2 = row * bh, (row + 1) * bh
            x1, x2 = col * bw, (col + 1) * bw
            b1 = g1[y1:y2, x1:x2]
            b2 = g2[y1:y2, x1:x2]
            score, _ = ssim(b1, b2, full=True)
            row_scores.append(round(float(score), 4))
            if score < 0.95:
                changed_blocks += 1
        block_scores.append(row_scores)

    total_blocks  = grid * grid
    changed_ratio = changed_blocks / total_blocks
    mean_ssim     = float(np.mean(block_scores))

    changed_positions = [
        (r, c)
        for r in range(grid)
        for c in range(grid)
        if block_scores[r][c] < 0.95
    ]
    if changed_positions:
        rows_affected  = len(set(r for r, _ in changed_positions))
        cols_affected  = len(set(c for _, c in changed_positions))
        is_distributed = (rows_affected >= grid // 2) and (cols_affected >= grid // 2)
    else:
        is_distributed = False

    return {
        "block_scores":   block_scores,
        "changed_ratio":  round(changed_ratio, 4),
        "is_distributed": is_distributed,
        "mean_ssim":      round(mean_ssim, 4),
    }


# ────────────────────────────────────────────
# 変化モード判定
# ────────────────────────────────────────────
def classify_change(scene_score: float, ssim_result: dict) -> str:
    """
    ffmpeg scene_score + SSIM ブロック分析から変化モードを判定する。

    判定マトリクス:
      荒い(>=5.0) & 全体的変化  → change
      荒い(>=5.0) & 局所的変化  → animation（フェードイン系）
      細かい      & 全体的変化  → change（じわじわ切り替わり）
      細かい      & 局所的変化  → animation
    """
    is_coarse      = scene_score >= SCENE_COARSE
    is_distributed = ssim_result["is_distributed"]
    changed_ratio  = ssim_result["changed_ratio"]

    if is_coarse and is_distributed:
        return "change"
    elif is_coarse and not is_distributed:
        return "animation"
    elif not is_coarse and changed_ratio >= SSIM_CHANGE_THRESHOLD:
        return "change"
    else:
        return "animation"


# ────────────────────────────────────────────
# OCR (LMStudio / GLM-OCR)
# ────────────────────────────────────────────
def run_ocr(image_path: str) -> str:
    """
    LMStudio の OpenAI 互換エンドポイントに画像を送り OCR 結果を返す。
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0,
    }

    try:
        resp = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] OCR 失敗 ({image_path}): {e}")
        return ""


# ────────────────────────────────────────────
# 動画長取得
# ────────────────────────────────────────────
def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[WARN] ffprobe で動画長取得失敗: {e}")
        return 0.0


# ────────────────────────────────────────────
# 時間範囲の計算
# ────────────────────────────────────────────
def compute_time_ranges(slides: list, video_duration: float):
    """
    各スライドに time（区間）を付与し、
    アニメーションフレームには time_range（区間）を付与する。
    detect_time（シーンチェンジ検知時刻）を境界として使用。
    """
    for i, slide in enumerate(slides):
        slide_start = slide["frames"][0]["detect_time"]
        slide_end   = slides[i + 1]["frames"][0]["detect_time"] if i + 1 < len(slides) else video_duration
        slide["time"] = {"start_sec": round(slide_start, 3), "end_sec": round(slide_end, 3)}

        frames = slide["frames"]
        for j, frame in enumerate(frames):
            if frame["mode"] == "animation":
                anim_start = frame["detect_time"]
                anim_end   = frames[j + 1]["detect_time"] if j + 1 < len(frames) else slide_end
                frame["time_range"] = {"start_sec": round(anim_start, 3), "end_sec": round(anim_end, 3)}


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────
def process_video(video_path: str, output_base: str | None = None):
    video_path = os.path.abspath(video_path)
    video_stem = Path(video_path).stem

    if output_base is None:
        output_base = str(Path(video_path).parent / video_stem)

    os.makedirs(output_base, exist_ok=True)
    print(f"[INFO] 動画: {video_path}")
    print(f"[INFO] 出力先: {output_base}")
    print(f"[INFO] OCRインターバル: {OCR_INTERVAL}秒")

    started_at = datetime.now()
    print(f"[INFO] 開始時刻: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step1: scene_score 取得 ──────────────────
    print("[INFO] scene_score 取得中...")
    frames = get_scene_scores_via_filter(video_path)

    if not frames:
        print("[ERROR] フレーム検知できませんでした。終了します。")
        return

    # SCENE_FINE 未満を足切り
    detected = [f for f in frames if f["score"] >= SCENE_FINE]
    print(f"[INFO] 処理対象フレーム数: {len(detected)}")

    if not detected:
        print("[WARN] 処理対象フレームなし。終了します。")
        return

    # ── Step2: フレーム分類 ──────────────────────
    slide_idx            = 1
    frame_idx            = 1
    slides               = []
    current_slide_frames = []
    last_ocr_time        = -OCR_INTERVAL  # 初回は必ずOCR実行されるように

    # 最初のフレームを slide1/picture1 として確定
    first_frame = extract_frame(video_path, detected[0]["time"])
    if first_frame is None:
        print("[ERROR] 最初のフレーム取得失敗")
        return

    slide_dir = os.path.join(output_base, f"slide{slide_idx}")
    os.makedirs(slide_dir, exist_ok=True)
    img_path  = os.path.join(slide_dir, f"picture{frame_idx}.jpg")
    cv2.imwrite(img_path, first_frame)

    print(f"[INFO] slide{slide_idx}/picture{frame_idx}  t={detected[0]['time']:.2f}s  (初期フレーム) → OCR実行中...")
    ocr_text      = run_ocr(img_path)
    last_ocr_time = detected[0]["time"]

    current_slide_frames.append({
        "id":          f"slide{slide_idx}_picture{frame_idx}",
        "time":        detected[0]["time"],
        "detect_time": detected[0]["time"],
        "mode":        "change",
        "image":       img_path,
        "ocr":         ocr_text,
    })

    prev_frame = first_frame
    frame_idx += 1

    for det in detected[1:]:
        curr_frame = extract_frame(video_path, det["time"])
        if curr_frame is None:
            continue

        # SSIM 分析
        ssim_result = ssim_block_analysis(prev_frame, curr_frame)
        mode        = classify_change(det["score"], ssim_result)

        # ── SSIM が高すぎる場合はスキップ（前フレームとほぼ同一）──
        if ssim_result["mean_ssim"] >= SSIM_SAME_THRESHOLD:
            print(
                f"  t={det['time']:.2f}s  scd={det['score']:.3f}"
                f"  ssim={ssim_result['mean_ssim']:.4f}"
                f"  → same  [SKIP: ssim >= {SSIM_SAME_THRESHOLD}]"
            )
            prev_frame = curr_frame
            continue

        # ── OCRインターバルチェック ──────────────
        elapsed = det["time"] - last_ocr_time
        if elapsed < OCR_INTERVAL:
            print(
                f"  t={det['time']:.2f}s  scd={det['score']:.3f}"
                f"  ssim={ssim_result['mean_ssim']:.4f}"
                f"  → {mode}  [SKIP: {elapsed:.2f}s < {OCR_INTERVAL}s]"
            )
            prev_frame = curr_frame
            continue

        print(
            f"  t={det['time']:.2f}s  scd={det['score']:.3f}"
            f"  ssim={ssim_result['mean_ssim']:.4f}"
            f"  changed={ssim_result['changed_ratio']:.2f}"
            f"  dist={ssim_result['is_distributed']}"
            f"  → {mode}"
        )

        if mode == "change":
            # 前スライドを確定
            if current_slide_frames:
                slides.append({
                    "slide_id":  f"slide{slide_idx}",
                    "frames":    current_slide_frames,
                    "full_text": current_slide_frames[-1]["ocr"],
                })
            slide_idx += 1
            frame_idx  = 1
            current_slide_frames = []

        slide_dir = os.path.join(output_base, f"slide{slide_idx}")
        os.makedirs(slide_dir, exist_ok=True)

        # 検知の瞬間でなく、OCR_DELAY秒待った後のフレームをOCR対象にする
        ocr_timestamp = det["time"] + OCR_DELAY
        ocr_frame     = extract_frame(video_path, ocr_timestamp)
        if ocr_frame is None:
            ocr_frame = curr_frame  # 動画末尾で取れない場合は検知フレームで代用

        img_path  = os.path.join(slide_dir, f"picture{frame_idx}.jpg")
        cv2.imwrite(img_path, ocr_frame)

        print(f"     → OCR実行中... t={ocr_timestamp:.2f}s ({img_path})")
        ocr_text      = run_ocr(img_path)
        last_ocr_time = det["time"]

        current_slide_frames.append({
            "id":          f"slide{slide_idx}_picture{frame_idx}",
            "time":        ocr_timestamp,
            "detect_time": det["time"],
            "mode":        mode,
            "image":       img_path,
            "ocr":         ocr_text,
        })

        prev_frame = curr_frame
        frame_idx += 1

    # 最後のスライドを確定
    if current_slide_frames:
        slides.append({
            "slide_id":  f"slide{slide_idx}",
            "frames":    current_slide_frames,
            "full_text": current_slide_frames[-1]["ocr"],
        })

    # ── Step3: 時間範囲の計算 ────────────────────
    video_duration = get_video_duration(video_path)
    compute_time_ranges(slides, video_duration)

    # ── Step4: JSON 出力 ─────────────────────────
    finished_at = datetime.now()
    elapsed     = (finished_at - started_at).total_seconds()

    output = {
        "meta": {
            "model":       LMSTUDIO_MODEL,
            "video":       video_path,
            "started_at":  started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 1),
        },
        "slides": slides,
    }

    json_path = os.path.join(output_base, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] スライド数: {len(slides)}")
    print(f"[DONE] JSON:  {json_path}")
    print(f"[DONE] 画像:  {output_base}/slide*/picture*.jpg")


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────
def main():
    global OCR_INTERVAL, LMSTUDIO_MODEL
    parser = argparse.ArgumentParser(description="スライド動画OCRツール")
    parser.add_argument("video",           help="動画ファイルパス (例: movie/test.mp4)")
    parser.add_argument("--output", "-o",  help="出力ディレクトリ（省略時は動画と同じ場所）")
    parser.add_argument("--interval", "-i", type=float, default=OCR_INTERVAL,
                        help=f"OCRインターバル秒数（デフォルト: {OCR_INTERVAL}）")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help=f"OCRモデル名（デフォルト: {LMSTUDIO_MODEL}）")
    args = parser.parse_args()

    if args.interval != OCR_INTERVAL:
        OCR_INTERVAL = args.interval
    if args.model:
        LMSTUDIO_MODEL = args.model

    process_video(args.video, args.output)


if __name__ == "__main__":
    main()
