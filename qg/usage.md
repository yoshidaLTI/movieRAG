# qg/ 使い方ガイド

講義動画の `result.json` から質問・解答を自動生成するモジュールです。  
全パターンの結果は `qg_result.json` 1ファイルに追記されます。

---

## 事前準備

- `extract/pipeline.py` を実行して `result.json` が生成済みであること
- LMStudio が起動していること（デフォルト: `http://localhost:1234`）
- パターン3・4を使う場合は事前に RAGインデックスを構築しておくこと

---

## 実行手順

### Step 1: RAGインデックスを構築（パターン3・4を使う場合）

```bash
python extract/rag_build.py movie/大分大学入門/result.json
```

### Step 2: 質問生成

```bash
# パターン1: スライド枚数基準・直接生成
python qg/qg_direct.py movie/大分大学入門/result.json --by-slides

# パターン2: 経過時間基準・直接生成
python qg/qg_direct.py movie/大分大学入門/result.json --by-time

# パターン3: スライド枚数基準・RAG
python qg/qg_rag.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-slides

# パターン4: 経過時間基準・RAG
python qg/qg_rag.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-time

# パターン5: スライド枚数基準・RAG + CoT
python qg/qg_rag_cot.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-slides

# パターン6: 経過時間基準・RAG + CoT
python qg/qg_rag_cot.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-time
```

結果は実行のたびに `movie/大分大学入門/qg_result.json` に追記されます。

---

## 6つのパターン

| パターン | 区切り方 | テキストの出所 | スクリプト |
|---|---|---|---|
| 1 | スライド枚数 | result.json 直接 | `qg_direct.py` |
| 2 | 経過時間 | result.json 直接 | `qg_direct.py` |
| 3 | スライド枚数 | RAG検索結果 | `qg_rag.py` |
| 4 | 経過時間 | RAG検索結果 | `qg_rag.py` |
| 5 | スライド枚数 | RAG検索結果 + CoT推論 | `qg_rag_cot.py` |
| 6 | 経過時間 | RAG検索結果 + CoT推論 | `qg_rag_cot.py` |

---

## パターン1・2：result.jsonから直接生成（qg_direct.py）

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--by-slides` | - | スライド枚数基準（パターン1）|
| `--by-time` | - | 経過時間基準（パターン2）|
| `--slides-per-chunk` | `3` | 何枚ごとに1チャンクとするか |
| `--minutes-per-chunk` | `5.0` | 何分ごとに1チャンクとするか |
| `--slide-overlap` | `0` | スライド基準の被り枚数（例: `1` → 隣チャンクと1枚重複） |
| `--time-overlap` | `0.0` | 時間基準の被り秒数（例: `60` → 隣チャンクと60秒重複） |
| `--questions-per-chunk` | `1` | 1チャンクあたりの生成問題数 |
| `--model` | `gpt-oss-20b` | LMStudioのモデル名 |
| `--output` | `qg_result.json` | 出力JSONパス（省略時: result.jsonと同ディレクトリ） |
| `--save-chunks` | - | LLMに渡すテキストを別ファイルに保存して確認する |
| `--lmstudio-url` | `http://localhost:1234/v1/chat/completions` | LMStudioエンドポイント |

---

## パターン3・4：RAGを使って生成（qg_rag.py）

### 処理の流れ

```
チャンク（OCR＋ASR）
  ↓ Step 1: seed抽出（LLM）
  キーワード（例: TCP/IP、OSI参照モデル）
  ↓ Step 2: RAG類似度検索
  関連講義資料（コンテキスト）
  ↓ Step 3: 質問・解答生成（LLM）
  question / answer / question_type / bloom_level
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--chroma` | `chroma_char_asr_ocr_mix` | RAGインデックスのパス（省略時: result.jsonと同ディレクトリのchroma_char_asr_ocr_mix） |
| `--by-slides` | - | スライド枚数基準（パターン3）|
| `--by-time` | - | 経過時間基準（パターン4）|
| `--slides-per-chunk` | `3` | 何枚ごとに1チャンクとするか |
| `--minutes-per-chunk` | `5.0` | 何分ごとに1チャンクとするか |
| `--slide-overlap` | `0` | スライド基準の被り枚数（例: `1` → 隣チャンクと1枚重複） |
| `--time-overlap` | `0.0` | 時間基準の被り秒数（例: `60` → 隣チャンクと60秒重複） |
| `--questions-per-chunk` | `1` | 1チャンクあたりの生成問題数 |
| `--model` | `gpt-oss-20b` | LMStudioのモデル名 |
| `--rag-k` | `3` | RAG検索で取得する上位件数 |
| `--embedding-model` | `cl-nagoya/ruri-v3-310m` | 埋め込みモデル名 |
| `--output` | `qg_result.json` | 出力JSONパス（省略時: result.jsonと同ディレクトリ） |
| `--save-chunks` | - | LLMに渡すテキスト・seedを別ファイルに保存して確認する |
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

---

## 出力JSONの形式（qg_result.json）

全パターン共通のフラットなリスト形式です。パターンを重ねて実行するたびに追記されます。

```json
[
  {
    "question_id":    "direct_slides_chunk_0000_q00",
    "model":          "gpt-oss-20b",
    "method":         "direct",
    "detail_setting": {
      "chunk_mode":          "slides",
      "slides_per_chunk":    3,
      "questions_per_chunk": 1,
      "bloom_level":         null
    },
    "chunk_id":  "chunk_0000",
    "slide_ids": ["slide1", "slide2", "slide3"],
    "start_sec": 0.0,
    "end_sec":   180.0,
    "question":  "TCPとUDPの主な違いはどれですか？",
    "choice_A":  "TCPは高速だがデータの到達を保証しない",
    "choice_B":  "UDPはコネクション確立が必要である",
    "choice_C":  "TCPは信頼性のある通信を保証し、UDPは保証しない",
    "choice_D":  "UDPはTCPよりも遅い",
    "answer":    "C",
    "reason":    "TCPは3ウェイハンドシェイクによりコネクションを確立し再送制御も行うため信頼性が高い。UDPはそれらを省略し低遅延を実現するが到達保証はない。",
    "bloom_level": "知識"
  },
  {
    "question_id":    "rag_cot_slides_chunk_0000_q00",
    "model":          "gpt-oss-20b",
    "method":         "rag_cot",
    "detail_setting": {
      "chunk_mode":          "slides",
      "slides_per_chunk":    3,
      "questions_per_chunk": 1,
      "bloom_level":         "評価",
      "rag_k":               3,
      "chroma":              "movie/大分大学入門/chroma_char_asr_ocr_mix"
    },
    "chunk_id":  "chunk_0000",
    "slide_ids": ["slide1", "slide2", "slide3"],
    "start_sec": 0.0,
    "end_sec":   180.0,
    "seeds":     ["TCP/IP", "OSI参照モデル", "パケット交換"],
    "cot":       "1. 主要概念: OSI参照モデルは...",
    "question":  "OSI参照モデルを採用する最大の利点はどれですか？",
    "choice_A":  "通信速度が向上する",
    "choice_B":  "異なるベンダー機器間の相互接続が容易になる",
    "choice_C":  "セキュリティが自動的に強化される",
    "choice_D":  "ハードウェアコストが削減される",
    "answer":    "B",
    "reason":    "OSI参照モデルは各層の役割を標準化することで、異なるメーカーの機器でも同じプロトコルで通信できるようにするための設計思想である。",
    "bloom_level": "評価"
  }
]
```

### 各フィールドの説明

| フィールド | 説明 |
|---|---|
| `question_id` | 質問の一意識別子（`{method}_{chunk_mode}_{chunk_id}_q{n}` 形式） |
| `model` | 質問生成に使用したLLMモデル名 |
| `method` | 生成方法（`direct` / `rag` / `rag_cot`） |
| `detail_setting` | チャンク設定・RAG設定の詳細 |
| `chunk_id` | チャンクの識別子 |
| `slide_ids` | このチャンクに含まれるスライドID一覧 |
| `start_sec` / `end_sec` | チャンクの時間範囲（秒） |
| `seeds` | RAGパターンのみ。LLMが抽出したキーワード |
| `cot` | パターン5・6のみ。LLMが生成した推論・分析テキスト |
| `question` | 生成された質問文 |
| `choice_A`〜`choice_D` | 4つの選択肢 |
| `answer` | 正解の選択肢（`A` / `B` / `C` / `D`） |
| `reason` | 正解の理由 |
| `bloom_level` | ブルームの教育目標分類（`知識` / `応用` / `評価`） |

---

## --save-chunks でLLMへの入力を確認する

```bash
# 直接生成パターン
python qg/qg_direct.py movie/大分大学入門/result.json --by-slides --save-chunks

# RAGパターン（seedとRAGコンテキストも保存）
python qg/qg_rag.py movie/大分大学入門/result.json \
    --chroma movie/大分大学入門/chroma_char_asr_ocr_mix \
    --by-slides --save-chunks
```

`qg_chunks_slides.json` が生成され、各チャンクでLLMに渡したテキストを確認できます。

---

## パターン5・6：RAG + CoT で生成（qg_rag_cot.py）

### 処理の流れ

```
チャンク（OCR＋ASR）
  ↓ Step 1: seed抽出（LLM）
  キーワード（例: TCP/IP、OSI参照モデル）
  ↓ Step 2: RAG類似度検索
  関連講義資料（コンテキスト）
  ↓ Step 3: CoT推論（LLM）
  主要概念・概念関係・つまずきポイント・出題可能事項の分析テキスト
  ↓ Step 4: 質問・解答生成（LLM）
  question / answer / question_type / bloom_level
```

パターン3・4と同じオプションが使用できます（`qg_rag.py` と同一インターフェース）。

出力 JSON に `"cot"` フィールドが追加されます。`--save-chunks` では `cot_text` も保存されます。

---

## モジュール構成

```
qg/
├── chunker.py        # スライド枚数・経過時間でのチャンク分割
├── seed_extractor.py # OCR+ASRからキーワード抽出（LLM）
├── qg_direct.py      # パターン1・2の実行スクリプト
├── qg_rag.py         # パターン3・4の実行スクリプト
└── qg_rag_cot.py     # パターン5・6の実行スクリプト（RAG + CoT）
```
