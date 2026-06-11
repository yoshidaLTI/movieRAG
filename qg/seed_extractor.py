# qg/seed_extractor.py

import json
import re
import requests

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

SEED_PROMPT = """\
以下の講義テキスト（スライドのOCRと音声書き起こし）から、質問を生成するための重要なキーワードやトピックを抽出してください。

[テキスト]
{text}

# 出力形式
以下のJSON形式のみで出力してください。前置き・説明・コードブロックは不要です。
最初の文字が {{ で最後の文字が }} であること。
{{"seeds": ["キーワード1", "キーワード2", ...]}}"""


def extract_seeds(
    ocr_text: str,
    asr_text: str,
    model: str,
    lmstudio_url: str = LMSTUDIO_URL,
    max_tokens: int = 512,
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

    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        data = json.loads(raw)
        seeds = data.get("seeds", [])
        if isinstance(seeds, list):
            return [str(s) for s in seeds if str(s).strip()]
    except json.JSONDecodeError:
        pass
    return []
