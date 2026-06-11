# qg/seed_extractor.py

import re
import requests

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

SEED_PROMPT = """\
以下の講義テキスト（スライドのOCRと音声書き起こし）から、質問を生成するための重要なキーワードやトピックを10個抽出してください。

[テキスト]
{text}

# 出力形式
キーワードを読点（、）区切りで出力してください。前置き・説明・コードブロックは不要です。
例: TCP/IP、OSI参照モデル、パケット交換"""


def extract_seeds(
    ocr_text: str,
    asr_text: str,
    model: str,
    lmstudio_url: str = LMSTUDIO_URL,
    max_tokens: int = 2048,
) -> list[str]:
    text_parts = []
    if ocr_text:
        text_parts.append(f"[スライド]\n{ocr_text}")
    if asr_text:
        text_parts.append(f"[音声]\n{asr_text}")
    text = "\n\n".join(text_parts)

    if not text.strip():
        return []

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": SEED_PROMPT.format(text=text)}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    try:
        resp = requests.post(lmstudio_url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] seed抽出失敗: {e}")
        return []

    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
    seeds = [s.strip() for s in re.split(r"[、,，\n]", raw) if s.strip()]
    return seeds
