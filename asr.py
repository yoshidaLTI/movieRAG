"""
asr.py

result.jsonの各スライドの時間範囲から音声を切り出し、
mlx-whisperでASRを実行してresult.jsonを更新する。

処理フロー:
  1. meta.proper_nouns_global を全スライド共通の initial_prompt として使用
  2. ffmpegでスライド区間の音声をWAVに切り出す
  3. mlx-whisperでASR
  4. セグメントのタイムスタンプを絶対時刻に変換
  5. result.jsonのasrフィールドを更新

使い方:
  python asr.py movie/test1.mp4 --result movie/test1/result.json
  python asr.py movie/test1.mp4 --result movie/test1/result.json --model mlx-community/whisper-large-v3-mlx
"""

import json
import argparse
import subprocess
import tempfile
from pathlib import Path

import mlx_whisper

# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────
DEFAULT_MODEL            = "mlx-community/whisper-large-v3-mlx"
LANGUAGE                 = "ja"
INITIAL_PROMPT_MAX_CHARS = 200  # 224トークン上限を考慮した文字数目安


# ────────────────────────────────────────────
# ffprobe で動画長を取得
# ────────────────────────────────────────────
def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


# ────────────────────────────────────────────
# 音声切り出し（ffmpeg）
# ────────────────────────────────────────────
def extract_audio_segment(video_path: str, start: float, end: float, out_path: str):
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", str(start), "-to", str(end),
         "-i", video_path,
         "-vn", "-ar", "16000", "-ac", "1",
         "-f", "wav", out_path],
        capture_output=True, check=True,
    )


# ────────────────────────────────────────────
# ASR（mlx-whisper）
# ────────────────────────────────────────────
def run_asr(audio_path: str, initial_prompt: str | None, model: str) -> list[dict]:
    kwargs = dict(
        path_or_hf_repo=model,
        language=LANGUAGE,
        word_timestamps=False,
        verbose=False,
        condition_on_previous_text=False,
    )
    if initial_prompt:
        kwargs["initial_prompt"]       = initial_prompt
        kwargs["carry_initial_prompt"] = True

    result = mlx_whisper.transcribe(audio_path, **kwargs)

    return [
        {"start_sec": round(seg["start"], 3),
         "end_sec":   round(seg["end"],   3),
         "text":      seg["text"].strip()}
        for seg in result.get("segments", [])
        if seg["text"].strip()
    ]


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────
def run(video_path: str, result_json_path: str, model: str):
    path = Path(result_json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    slides = data.get("slides", [])
    if not slides:
        print("[WARN] スライドが存在しません。終了します。")
        return

    # 全スライド共通の initial_prompt を meta から取得
    global_nouns = data.get("meta", {}).get("proper_nouns_global", [])
    if not global_nouns:
        print("[WARN] meta.proper_nouns_global が存在しません。")
        print("       先に proper_nouns_global.py を実行してください。")
        return

    initial_prompt = "、".join(global_nouns)[:INITIAL_PROMPT_MAX_CHARS]
    print(f"[INFO] InitialPrompt ({len(global_nouns)}語): {initial_prompt}")

    video_duration = get_video_duration(video_path)
    print(f"[INFO] 動画長: {video_duration:.1f}秒  スライド数: {len(slides)}  モデル: {model}")

    if "time" in slides[0]:
        slide_ranges = [(s["time"]["start_sec"], s["time"]["end_sec"]) for s in slides]
    else:
        # time フィールドがない場合は frames[0].time から計算
        slide_ranges = []
        for i, slide in enumerate(slides):
            start = slide["frames"][0]["time"]
            end   = slides[i+1]["frames"][0]["time"] if i+1 < len(slides) else video_duration
            slide_ranges.append((start, end))

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (slide, (start, end)) in enumerate(zip(slides, slide_ranges)):
            slide_id = slide.get("slide_id", f"slide{i+1}")
            duration = end - start

            if duration <= 0:
                print(f"  {slide_id}: 区間長0秒 → スキップ")
                slide["asr"] = []
                continue

            print(f"  {slide_id}: {start:.1f}s〜{end:.1f}s ({duration:.1f}秒) 処理中...", end=" ", flush=True)

            wav_path = f"{tmpdir}/{slide_id}.wav"
            try:
                extract_audio_segment(video_path, start, end, wav_path)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] 音声切り出し失敗: {e}")
                slide["asr"] = []
                continue

            segments = run_asr(wav_path, initial_prompt, model)

            for seg in segments:
                seg["start_sec"] = round(seg["start_sec"] + start, 3)
                seg["end_sec"]   = round(seg["end_sec"]   + start, 3)

            slide["asr"] = segments
            print(f"{len(segments)}セグメント取得")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {path} を更新しました。")


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="スライド単位でASRを実行してresult.jsonに追記")
    parser.add_argument("video", help="動画ファイルパス (例: movie/test1.mp4)")
    parser.add_argument("--result", "-r",
                        help="result.jsonのパス（省略時: <動画の親>/<動画stem>/result.json）")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"mlx-whisperモデル (デフォルト: {DEFAULT_MODEL})")
    args = parser.parse_args()

    result_json = args.result or str(Path(args.video).parent / Path(args.video).stem / "result.json")
    run(args.video, result_json, args.model)


if __name__ == "__main__":
    main()
