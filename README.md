# movieRAG

講義動画（.mp4）から**スライドのテキスト（OCR）・音声（ASR）を自動抽出**し、質問応答に使える**RAG（検索インデックス）**を構築するツール群です。

> **OCR**：画像からテキストを読み取る技術  
> **ASR**：音声をテキストに書き起こす技術（自動音声認識）  
> **RAG**：質問に関連する文章を検索して回答に活用する仕組み

---

## 必要なもの

| ツール | 用途 | インストール方法 |
|---|---|---|
| Python 3.11以上 | スクリプト実行 | [pyenv](https://github.com/pyenv/pyenv) などで導入 |
| ffmpeg | 動画からフレーム抽出 | `brew install ffmpeg`（Mac） |
| [LMStudio](https://lmstudio.ai/) | OCR・固有名詞抽出モデルの実行 | 公式サイトからダウンロード |

LMStudio には以下のモデルをロードして起動しておいてください（`http://localhost:1234` で待機）：
- OCRモデル（例: `glm-ocr`）
- 固有名詞抽出モデル（例: `gpt-oss-20b`）

---

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

# 4. ffmpeg をインストール（macOS）
brew install ffmpeg
```

---

## ディレクトリ構成

```
movieRAG/
├── extract/          # OCR・ASR・RAG構築スクリプト
├── qg/               # 質問生成（今後実装予定）
├── eval/             # 評価スクリプト・データ
├── rag/              # RAG共通モジュール
├── slide_base_rag/   # スライド単位RAG共通モジュール
└── movie/            # 動画ファイルを置く場所
```

---

## 使い方

### 動画ファイルを置く

処理したい動画を `movie/` フォルダに入れます。

```
movie/
└── lecture.mp4   ← ここに置く（ファイル名は自由）
```

---

### パイプライン全体図

```
lecture.mp4（講義動画）
      │
      ▼ Step 1
      スライドの切り替わりを検知してOCRでテキスト化
      → movie/lecture/result.json
      → movie/lecture/slide1/picture1.jpg（スライド画像）
      │
      ▼ Step 2
      全スライドのOCRから固有名詞リストを作成（ASRの精度向上のため）
      → result.json の meta.proper_nouns_global に追記
      │
      ▼ Step 3
      固有名詞リストをヒントに音声をテキスト化（ASR）
      → result.json の各スライドの asr フィールドに追記
      │
      ├─▶ Step 4-A  文字数ベースでRAGインデックスを構築
      │             → movie/lecture/chroma_char_*/
      └─▶ Step 4-B  スライド単位でRAGインデックスを構築
                    → movie/lecture/chroma_slide_*/
```

---

### Step 1〜3：一括実行

```bash
python extract/pipeline.py movie/lecture.mp4 \
    --ocr-model   glm-ocr \
    --nouns-model gpt-oss-20b \
    --asr-model   mlx-community/whisper-large-v3-mlx
```

実行後、`movie/lecture/` フォルダが自動で作成され、スライド画像と `result.json` が生成されます。

#### 途中からやり直す場合

```bash
# Step 1（OCR）はスキップして Step 2以降だけ実行
python extract/pipeline.py movie/lecture.mp4 --skip-ocr

# Step 1・2をスキップして Step 3（ASR）だけ実行
python extract/pipeline.py movie/lecture.mp4 --skip-ocr --skip-nouns
```

#### 各Stepを個別に実行する場合

```bash
# Step 1: スライド検知 + OCR
python extract/slide_checker.py movie/lecture.mp4 --model glm-ocr

# Step 2: 固有名詞リスト作成
python extract/proper_nouns_global.py movie/lecture/result.json --model gpt-oss-20b

# Step 3: 音声書き起こし（ASR）
python extract/asr.py movie/lecture.mp4 \
    --result movie/lecture/result.json \
    --model mlx-community/whisper-large-v3-mlx
```

---

### Step 4-A：文字数ベースRAGを構築・検索

```bash
# インデックス構築
python extract/rag_build.py movie/lecture/result.json
```

以下の3種類のインデックスが作成されます：

| ディレクトリ | 内容 |
|---|---|
| `movie/lecture/chroma_char_asr_ocr_mix/` | OCR+ASR 混合 |
| `movie/lecture/chroma_char_ocr/` | OCR のみ |
| `movie/lecture/chroma_char_asr/` | ASR のみ |

```bash
# 検索
python extract/rag_search.py "この授業のテーマは？" --chroma movie/lecture/chroma_char_asr_ocr_mix
```

---

### Step 4-B：スライド単位RAGを構築・検索

```bash
# インデックス構築
python extract/rag_build_slide.py movie/lecture/result.json
```

| ディレクトリ | 内容 |
|---|---|
| `movie/lecture/chroma_slide_asr_ocr_mix/` | OCR+ASR 混合 |
| `movie/lecture/chroma_slide_ocr/` | OCR のみ |
| `movie/lecture/chroma_slide_asr/` | ASR のみ |

```bash
# 検索
python extract/rag_search_slide.py "この授業のテーマは？" --chroma movie/lecture/chroma_slide_asr_ocr_mix
```

---

### ユーティリティ

```bash
# result.json からASRテキストだけを取り出して確認
python extract/extract_asr.py movie/lecture/result.json
```

---

## result.json の構造

各Stepの結果はすべて `result.json` に蓄積されます。

```json
{
  "meta": {
    "model": "glm-ocr",
    "video": "movie/lecture.mp4",
    "proper_nouns_global": ["TCP/IP", "OSI参照モデル", "ARPANET"]
  },
  "slides": [
    {
      "slide_id": "slide1",
      "time": { "start_sec": 0.4, "end_sec": 19.64 },
      "frames": [
        {
          "id": "slide1_picture1",
          "time": 0.4,
          "mode": "change",
          "image": "movie/lecture/slide1/picture1.jpg",
          "ocr": "スライドのテキスト..."
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

## RAGインデックスの種類について

2種類のRAG構築方法があります。用途に合わせて選んでください。

| | 文字数ベース（char） | スライド単位（slide） |
|---|---|---|
| 向いている用途 | 長い音声の内容を検索したい | スライドの内容で検索したい |
| 親チャンクの区切り | ASRテキスト1000文字ごと | 3スライドまとめて1ブロック |
| 子チャンク | 親を300文字に細分化 | フレーム（change/animation）単位 |
| 子の内容 | ASR（なければOCR） | そのフレームのOCR＋該当区間のASR |

---

## オプション一覧

### extract/pipeline.py

| オプション | デフォルト | 説明 |
|---|---|---|
| `--ocr-model` | `glm-ocr` | OCRモデル名（LMStudioにロード済みのもの） |
| `--nouns-model` | `gpt-oss-20b` | 固有名詞抽出モデル名 |
| `--asr-model` | `mlx-community/whisper-large-v3-mlx` | ASRモデル名 |
| `--skip-ocr` | - | Step1をスキップ |
| `--skip-nouns` | - | Step2をスキップ |
| `--skip-asr` | - | Step3をスキップ |

### extract/rag_build.py

| オプション | デフォルト | 説明 |
|---|---|---|
| `--min-asr-chars` | 1000 | 親チャンクの最小ASR文字数 |
| `--child-chunk-size` | 300 | 子チャンクの文字数 |
| `--child-chunk-overlap` | 30 | 子チャンクのオーバーラップ文字数 |
| `--no-hint` | - | 固有名詞ヒントを使わない |

### extract/rag_build_slide.py

| オプション | デフォルト | 説明 |
|---|---|---|
| `--group-size` | 3 | 1つの親チャンクにまとめるスライド枚数 |
| `--no-hint` | - | 固有名詞ヒントを使わない |
