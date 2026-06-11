# movieRAG

動画からスライドの変化・アニメーションを検知し、OCR・ASR・RAGインデックスを構築するツール群。

## セットアップ

```bash
# 1. リポジトリを取得
git clone https://github.com/yoshidaLTI/movieRAG.git
cd movieRAG

# 2. 仮想環境を作成して有効化
python -m venv .venv
source .venv/bin/activate   # Mac/Linux

# 3. 依存ライブラリをインストール
pip install -r requirements.txt

# 4. ffmpeg（macOS）
brew install ffmpeg

# 5. LMStudio を起動し、以下のモデルをロードしておく
#   - OCRモデル（例: glm-ocr）
#   - 固有名詞抽出モデル（例: gpt-oss-20b）
#   http://localhost:1234 で起動していることを確認
```

---

## パイプライン全体図

```
動画ファイル (.mp4)
      │
      ▼
① slide_checker.py       スライド検知 + OCR
      │                  → result.json（frames, full_text, time）
      ▼
② proper_nouns_global.py  動画全体から共通固有名詞リストを作成
      │                  → result.json の meta.proper_nouns_global に追記
      ▼
③ asr.py                 meta.proper_nouns_global を InitialPrompt として ASR
      │                  → result.json の asr フィールドに追記
      ▼
④-A rag_build.py          文字数チャンキング RAG 構築 → chroma_char_*/
④-B rag_build_slide.py    スライド単位 RAG 構築      → chroma_slide_*/
```

---

## Step 1〜3：一括実行（pipeline.py）

```bash
python extract/pipeline.py movie/test1.mp4 \
    --ocr-model   glm-ocr \
    --nouns-model gpt-oss-20b \
    --asr-model   mlx-community/whisper-large-v3-mlx
```

### 個別実行

```bash
# Step 1: スライド検知 + OCR
python extract/slide_checker.py movie/test1.mp4 --model glm-ocr

# Step 2: 動画全体の共通固有名詞リスト作成
python extract/proper_nouns_global.py movie/test1/result.json --model gpt-oss-20b

# Step 3: ASR（meta.proper_nouns_global を InitialPrompt として使用）
python extract/asr.py movie/test1.mp4 --result movie/test1/result.json \
    --model mlx-community/whisper-large-v3-mlx
```

### 途中から再実行

```bash
# Step 2以降のみ
python extract/pipeline.py movie/test1.mp4 --skip-ocr

# Step 3のみ
python extract/pipeline.py movie/test1.mp4 --skip-ocr --skip-nouns
```

---

## Step 4-A：文字数チャンキング RAG

```bash
python extract/rag_build.py movie/test1/result.json
```

| ディレクトリ | 内容 |
|---|---|
| `movie/test1/chroma_char_asr_ocr_mix/` | OCR+ASR 混合 |
| `movie/test1/chroma_char_ocr/` | OCR のみ |
| `movie/test1/chroma_char_asr/` | ASR のみ |

```bash
python extract/rag_search.py "TCP/IPとは？" --chroma movie/test1/chroma_char_asr_ocr_mix
```

---

## Step 4-B：スライド単位 RAG

```bash
python extract/rag_build_slide.py movie/test1/result.json
```

| ディレクトリ | 内容 |
|---|---|
| `movie/test1/chroma_slide_asr_ocr_mix/` | OCR+ASR 混合 |
| `movie/test1/chroma_slide_ocr/` | OCR のみ |
| `movie/test1/chroma_slide_asr/` | ASR のみ |

```bash
python extract/rag_search_slide.py "TCP/IPとは？" --chroma movie/test1/chroma_slide_asr_ocr_mix
```

---

## テキスト抽出ユーティリティ

```bash
# ASR テキストを抽出
python extract/extract_asr.py movie/test1/result.json
```

---

## result.json の構造

```json
{
  "meta": {
    "model": "glm-ocr",
    "video": "...",
    "proper_nouns_global": ["TCP/IP", "OSI参照モデル", "ARPANET", ...]
  },
  "slides": [
    {
      "slide_id": "slide1",
      "time": { "start_sec": 0.4, "end_sec": 19.64 },
      "frames": [
        {
          "id": "slide1_picture1",
          "time": 0.4,
          "detect_time": 0.4,
          "mode": "change",
          "image": "movie/test1/slide1/picture1.jpg",
          "ocr": "スライドのテキスト..."
        },
        {
          "id": "slide1_picture2",
          "time": 108.04,
          "detect_time": 107.04,
          "mode": "animation",
          "time_range": { "start_sec": 107.04, "end_sec": 122.16 },
          "ocr": "アニメーション後のテキスト..."
        }
      ],
      "full_text": "最終フレームのOCRテキスト",
      "asr": [
        { "start_sec": 0.4, "end_sec": 9.0, "text": "講義を始めます..." }
      ]
    }
  ]
}
```

---

## RAG チャンキング戦略の比較

| | 文字数ベース | スライドベース |
|---|---|---|
| 親チャンク境界 | ASR文字数1000文字で区切り | 3スライドずつ固定 |
| 子チャンク | 親を300文字に分割 | フレーム（change/animation）単位 |
| 子の内容 | ASR（なければOCR） | そのフレームのOCR＋時間範囲内のASR |

---

## パラメータ一覧

### pipeline.py

| オプション | デフォルト | 説明 |
|---|---|---|
| `--ocr-model` | `glm-ocr` | OCRモデル名 |
| `--nouns-model` | `gpt-oss-20b` | 固有名詞抽出モデル名（動画全体共通リスト） |
| `--asr-model` | `mlx-community/whisper-large-v3-mlx` | ASRモデル名 |
| `--skip-ocr` | - | Step1をスキップ |
| `--skip-nouns` | - | Step2をスキップ |
| `--skip-asr` | - | Step3をスキップ |

### rag_build.py

| オプション | デフォルト | 説明 |
|---|---|---|
| `--min-asr-chars` | 1000 | 親チャンクの最小ASR文字数 |
| `--child-chunk-size` | 300 | 子チャンクの文字数 |
| `--child-chunk-overlap` | 30 | 子チャンクのオーバーラップ |
| `--no-hint` | - | proper_nounsをヒントに使わない |

### rag_build_slide.py

| オプション | デフォルト | 説明 |
|---|---|---|
| `--group-size` | 3 | 親チャンクにまとめるスライド枚数 |
| `--no-hint` | - | proper_nounsをヒントに使わない |
