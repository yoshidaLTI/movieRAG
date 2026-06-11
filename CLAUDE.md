# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Python dependencies
pip install -r requirements.txt

# System dependency (required for scdet filter and audio extraction)
brew install ffmpeg   # macOS
# sudo apt install ffmpeg  # Ubuntu

# LMStudio must be running locally with the required models loaded
# Default endpoint: http://localhost:1234
# Required models: OCR model (e.g. glm-ocr), noun extraction model (e.g. gpt-oss-20b)
```

## Running

The recommended entry point is `pipeline.py`, which runs steps 1–3 in sequence:

```bash
python pipeline.py movie/test1.mp4 \
    --ocr-model   glm-ocr \
    --nouns-model gpt-oss-20b \
    --asr-model   mlx-community/whisper-large-v3-mlx

# Skip already-completed steps
python pipeline.py movie/test1.mp4 --skip-ocr            # skip slide detection + OCR
python pipeline.py movie/test1.mp4 --skip-ocr --skip-nouns  # ASR only
```

Individual scripts can also be run directly:

```bash
# Step 1: slide detection + OCR → writes result.json and slideN/pictureM.jpg
python slide_checker.py movie/test1.mp4 --model glm-ocr

# Step 2: extract proper nouns from all OCR text → updates meta.proper_nouns_global
python proper_nouns_global.py movie/test1/result.json --model gpt-oss-20b

# Step 3: ASR per slide, using proper nouns as Whisper InitialPrompt
python asr.py movie/test1.mp4 --result movie/test1/result.json \
    --model mlx-community/whisper-large-v3-mlx

# Step 4-A: character-count chunking RAG index → chroma_char_*/
python rag_build.py movie/test1/result.json

# Step 4-B: slide-unit RAG index → chroma_slide_*/
python rag_build_slide.py movie/test1/result.json

# Search
python rag_search.py "TCP/IPとは？" --chroma movie/test1/chroma_char_asr_ocr_mix
python rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_asr_ocr_mix

# Utility: extract ASR text from result.json
python extract_asr.py movie/test1/result.json
```

There are no tests or linting configured in this project.

## Evaluation

```bash
# Evaluate RAG quality vs human annotations (Spearman correlation across 4 RAG configs × 2 LLMs)
python eval_rag_quality.py
python eval_rag_quality.py --models gpt-oss-20b qwen/qwen3-vl-30b
python eval_rag_quality.py --video 31   # single video only
# Output → previous_qg/rag_quality_scores.json, previous_qg/rag_quality_result.json

# Dump per-question accuracy scores from a stored eval JSON
python dump_qwen_scores.py
```

## Architecture

### Pipeline overview

```
video (.mp4)
  │
  ▼ slide_checker.py        scene detection + OCR
  │                         → result.json  +  slideN/pictureM.jpg
  ▼ proper_nouns_global.py  LMStudio extracts proper nouns from all OCR text
  │                         → result.json meta.proper_nouns_global
  ▼ asr.py                  mlx-whisper ASR per slide, using proper nouns as InitialPrompt
  │                         → result.json slides[].asr
  ├─▶ rag_build.py          character-count chunked RAG  → chroma_char_*/
  └─▶ rag_build_slide.py    slide-unit RAG               → chroma_slide_*/
```

All intermediate and final state lives in `result.json`. Each step reads and updates this file.

### slide_checker.py (Step 1)

The detection pipeline per frame:

1. **`get_scene_scores_via_filter`** — shells out to `ffmpeg -vf scdet` to get `lavfi.scd.score` and `lavfi.scd.time`. Frames below `SCENE_FINE` (0.005) are discarded.
2. **`ssim_block_analysis`** — splits consecutive frames into a `SSIM_GRID×SSIM_GRID` (4×4) grid and computes per-block SSIM. Returns `changed_ratio` and `is_distributed`.
3. **`classify_change`** — combines ffmpeg score + SSIM to label each frame:
   - `score >= SCENE_COARSE` (5.0) + distributed → `"change"` (new slide)
   - `score >= SCENE_COARSE` + localized → `"animation"`
   - `score < SCENE_COARSE` + `changed_ratio >= SSIM_CHANGE_THRESHOLD` (0.6) → `"change"`
   - otherwise → `"animation"`
4. **`run_ocr`** — sends the frame as base64 to LMStudio (`/v1/chat/completions`). OCR is skipped if the previous call was within `OCR_INTERVAL` (1.5s). The captured frame is taken `OCR_DELAY` (1.0s) after the detected timestamp to let animations settle.
5. **`process_video`** — orchestrates the above, writes `slideN/pictureM.jpg` and `result.json`.

### slide_base_rag/ (alternative RAG module)

A second RAG implementation in `slide_base_rag/` with `indexer.py` and `retriever.py`. It shares `rag/embeddings.py` for embeddings but has its own indexing/retrieval logic. Used by `eval_rag_quality.py` for evaluation comparisons.

### RAG module (`rag/`)

- `rag/embeddings.py` — wraps `cl-nagoya/ruri-v3-310m` (default embedding model)
- `rag/indexer.py` — builds parent/child ChromaDB collections from `result.json`
- `rag/retriever.py` — loads collections and performs parent-document retrieval

Both `rag_build.py` and `rag_build_slide.py` call `rag/indexer.py`; they differ only in chunking strategy:

| | `rag_build.py` (char) | `rag_build_slide.py` (slide) |
|---|---|---|
| Parent chunk | ASR text, ≥1000 chars | 3 slides fixed |
| Child chunk | 300-char splits of parent | per frame (change/animation) |
| Child content | ASR (fallback OCR) | frame OCR + overlapping ASR |

### result.json structure

```json
{
  "meta": {
    "model": "glm-ocr",
    "video": "movie/test1.mp4",
    "proper_nouns_global": ["TCP/IP", "OSI参照モデル", ...]
  },
  "slides": [
    {
      "slide_id": "slide1",
      "time": { "start_sec": 0.4, "end_sec": 19.64 },
      "full_text": "最終フレームのOCRテキスト",
      "frames": [
        {
          "id": "slide1_picture1",
          "time": 0.4,
          "detect_time": 0.4,
          "mode": "change",
          "image": "movie/test1/slide1/picture1.jpg",
          "ocr": "..."
        }
      ],
      "asr": [
        { "start_sec": 0.4, "end_sec": 9.0, "text": "..." }
      ]
    }
  ]
}
```

## Tunable parameters

Constants at the top of each script:

**`slide_checker.py`**

| Constant | Default | Purpose |
|---|---|---|
| `SCENE_COARSE` | 5.0 | ffmpeg score threshold for obvious scene change |
| `SCENE_FINE` | 0.005 | Minimum score to process a frame at all |
| `SSIM_GRID` | 4 | Grid size for block SSIM (4×4 = 16 blocks) |
| `SSIM_CHANGE_THRESHOLD` | 0.6 | Fraction of changed blocks → `change` mode |
| `SSIM_SAME_THRESHOLD` | 0.99 | Mean SSIM above this → frame treated as identical, skipped |
| `OCR_INTERVAL` | 1.5 | Min seconds between OCR calls |
| `OCR_DELAY` | 1.0 | Seconds after detection to capture the OCR frame |
| `LMSTUDIO_URL` | `http://localhost:1234/v1/chat/completions` | LMStudio endpoint |
| `LMSTUDIO_MODEL` | `"glm-ocr"` | Model name as configured in LMStudio |

**`asr.py`**

| Constant | Default | Purpose |
|---|---|---|
| `LANGUAGE` | `"ja"` | Whisper language code |
| `INITIAL_PROMPT_MAX_CHARS` | 200 | Truncation limit for proper-noun prompt |

**`rag_build.py` / `rag_build_slide.py`** — all tunable via CLI flags (see `--help`).

## Evaluation data

`previous_qg/` — human evaluation answers and prior auto-eval results used as ground truth in `eval_rag_quality.py`.  
`claude_knowledge/` — cached LLM evaluation outputs (e.g. `RAG-EVAL_QWEN-3-VL30b.json`) for offline analysis with `dump_qwen_scores.py`.
