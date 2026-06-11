# rag/embeddings.py

from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings


class RuriEmbeddings(Embeddings):
    """
    cl-nagoya/ruri-v3-310m をローカルで実行する日本語埋め込みモデル。
    Apple Silicon MPS / CUDA / CPU に対応。
    """

    def __init__(self, model_name: str = "cl-nagoya/ruri-v3-310m"):
        self.device = (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()         else
            "cpu"
        )
        print(f"[embeddings] デバイス: {self.device}")
        print(f"[embeddings] モデルロード中: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print("[embeddings] モデルロード完了")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        # Mean pooling
        mask   = encoded["attention_mask"]
        token_emb = outputs.last_hidden_state
        expanded  = mask.unsqueeze(-1).expand(token_emb.size()).float()
        emb = torch.sum(token_emb * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """ドキュメント用（ruri は "passage: " プレフィックスを使用）"""
        return self._embed([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        """クエリ用（ruri は "query: " プレフィックスを使用）"""
        return self._embed([f"query: {text}"])[0]
