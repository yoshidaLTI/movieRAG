# qg/ 使い方ガイド

講義動画の `result.json` から質問・解答を自動生成するモジュールです。

---

## 事前準備

- `extract/pipeline.py` を実行して `result.json` が生成済みであること
- LMStudio が起動していること（デフォルト: `http://localhost:1234`）
- パターン3・4を使う場合は `extract/rag_build.py` でRAGインデックスが構築済みであること

---

## 4つのパターン

| パターン | 区切り方 | テキストの出所 | スクリプト |
|---|---|---|---|
| 1 | スライド枚数 | result.json 直接 | `qg_direct.py` |
| 2 | 経過時間 | result.json 直接 | `qg_direct.py` |
| 3 | スライド枚数 | RAG検索結果 | `qg_rag.py` |
| 4 | 経過時間 | RAG検索結果 | `qg_rag.py` |

---

## パターン1・2：result.jsonから直接生成（qg_direct.py）

### パターン1：スライド枚数基準

```bash
python qg/qg_direct.py movie/lecture/result.json --by-slides
```

### パターン2：経過時間基準

```bash
python qg/qg_direct.py movie/lecture/result.json --by-time
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--by-slides` | - | スライド枚数基準（パターン1）|
| `--by-time` | - | 経過時間基準（パターン2）|
| `--slides-per-chunk` | `3` | 何枚ごとに1問生成するか |
| `--minutes-per-chunk` | `5.0` | 何分ごとに1問生成するか |
| `--model` | `qwen/qwen3-vl-8b` | LMStudioのモデル名 |
| `--output` | 自動 | 出力JSONパス |
| `--lmstudio-url` | `http://localhost:1234/v1/chat/completions` | LMStudioエンドポイント |

### 出力ファイル

省略時は `result.json` と同じディレクトリに自動で保存されます。

| パターン | デフォルトの出力ファイル |
|---|---|
| パターン1 | `movie/lecture/qg_direct_slides.json` |
| パターン2 | `movie/lecture/qg_direct_time.json` |

---

## パターン3・4：RAGを使って生成（qg_rag.py）

### 処理の流れ

```
チャンク（OCR＋ASR）
  ↓ Step 1: seed抽出（LLM）
  キーワード（例: TCP/IP、OSI参照モデル）
  ↓ Step 2: RAG類似度検索
  関連講義資料
  ↓ Step 3: 質問・解答生成（LLM）
  question / answer
```

### パターン3：スライド枚数基準

```bash
python qg/qg_rag.py movie/lecture/result.json \
    --chroma movie/lecture/chroma_char_asr_ocr_mix \
    --by-slides
```

### パターン4：経過時間基準

```bash
python qg/qg_rag.py movie/lecture/result.json \
    --chroma movie/lecture/chroma_char_asr_ocr_mix \
    --by-time
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--chroma` | 必須 | RAGインデックスのパス |
| `--by-slides` | - | スライド枚数基準（パターン3）|
| `--by-time` | - | 経過時間基準（パターン4）|
| `--slides-per-chunk` | `3` | 何枚ごとに1問生成するか |
| `--minutes-per-chunk` | `5.0` | 何分ごとに1問生成するか |
| `--model` | `qwen/qwen3-vl-8b` | LMStudioのモデル名 |
| `--rag-k` | `3` | RAG検索で取得する上位件数 |
| `--embedding-model` | `cl-nagoya/ruri-v3-310m` | 埋め込みモデル名 |
| `--output` | 自動 | 出力JSONパス |
| `--lmstudio-url` | `http://localhost:1234/v1/chat/completions` | LMStudioエンドポイント |

### 使用できるRAGインデックス

| インデックス | 特徴 |
|---|---|
| `chroma_char_asr_ocr_mix` | 文字数ベース・OCR＋ASR混合（推奨） |
| `chroma_char_asr` | 文字数ベース・ASRのみ |
| `chroma_char_ocr` | 文字数ベース・OCRのみ |
| `chroma_slide_asr_ocr_mix` | スライド単位・OCR＋ASR混合 |
| `chroma_slide_asr` | スライド単位・ASRのみ |
| `chroma_slide_ocr` | スライド単位・OCRのみ |

### 出力ファイル

| パターン | デフォルトの出力ファイル |
|---|---|
| パターン3 | `movie/lecture/qg_rag_slides.json` |
| パターン4 | `movie/lecture/qg_rag_time.json` |

---

## 出力JSONの形式

### パターン1・2（qg_direct）

```json
[
  {
    "chunk_id": "chunk_0000",
    "slide_ids": ["slide1", "slide2", "slide3"],
    "start_sec": 0.0,
    "end_sec": 180.0,
    "question": "TCPとUDPの違いを説明してください。",
    "answer": "TCPは信頼性のある通信を保証するプロトコルで..."
  }
]
```

### パターン3・4（qg_rag）

```json
[
  {
    "chunk_id": "chunk_0000",
    "slide_ids": ["slide1", "slide2", "slide3"],
    "start_sec": 0.0,
    "end_sec": 180.0,
    "seeds": ["TCP/IP", "OSI参照モデル", "パケット交換"],
    "question": "OSI参照モデルの第4層（トランスポート層）の役割は何ですか？",
    "answer": "トランスポート層はエンドツーエンドの通信を管理し..."
  }
]
```

---

## 使用例（大分大学入門）

```bash
# RAGインデックスを先に構築
python extract/rag_build.py movie/大分大学入門/result.json

# パターン1: スライド3枚ごとに直接生成
python qg/qg_direct.py movie/大分大学入門/result.json --by-slides

# パターン2: 5分ごとに直接生成
python qg/qg_direct.py movie/大分大学入門/result.json --by-time

# パターン3: スライド3枚ごと + RAG
python qg/qg_rag.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-slides

# パターン4: 5分ごと + RAG
python qg/qg_rag.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-time
```

---

## モジュール構成

```
qg/
├── chunker.py        # スライド枚数・経過時間でのチャンク分割
├── seed_extractor.py # OCR+ASRからキーワード抽出（LLM）
├── qg_direct.py      # パターン1・2の実行スクリプト
└── qg_rag.py         # パターン3・4の実行スクリプト
```
