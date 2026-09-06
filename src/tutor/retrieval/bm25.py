"""轻量 BM25 检索,零外部依赖。

选它而不是向量库的原因:
- 这个场景的语料是**单份材料**(几十到几百个 chunk),BM25 的召回足够;
- 不引入 embedding 服务 = 少一个 API Key、少一层部署、结果可复现;
- 对中文用「字符 bigram + 英文词」的混合分词,不依赖 jieba 之类的词典。

接口留成 `Retriever` 协议,想换成向量检索只要实现同样的 search()。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from ..schemas import Chunk, RetrievedChunk

K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]*|\d+(?:\.\d+)?")
_CJK = re.compile(r"[一-鿿]")
_STOP = {"the", "a", "an", "of", "is", "are", "to", "and", "in", "for", "on", "it"}


def tokenize(text: str) -> list[str]:
    """英文按词、数字成 token;中文取连续汉字串的 unigram + bigram。"""
    lowered = text.lower()
    tokens = [t for t in _WORD.findall(lowered) if t not in _STOP]
    for run in re.findall(r"[一-鿿]+", lowered):
        tokens.extend(run)  # unigram:保证短词也能召回
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigram:提升精度
    return tokens


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._docs: list[Counter[str]] = []
        self._lengths: list[int] = []
        df: Counter[str] = Counter()

        for chunk in chunks:
            # 小节标题也参与索引:标题往往就是知识点名
            tokens = tokenize(f"{chunk.section} {chunk.text}")
            counts = Counter(tokens)
            self._docs.append(counts)
            self._lengths.append(max(1, len(tokens)))
            df.update(counts.keys())

        n = max(1, len(chunks))
        self._avg_len = sum(self._lengths) / n
        # BM25 的 idf(加 0.5 平滑,再取 max 防负值)
        self._idf = {
            term: max(1e-6, math.log(1 + (n - freq + 0.5) / (freq + 0.5)))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_terms = tokenize(query)
        if not q_terms or not self.chunks:
            return []

        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._docs):
            score = 0.0
            length = self._lengths[i]
            for term in q_terms:
                tf = counts.get(term)
                if not tf:
                    continue
                denom = tf + K1 * (1 - B + B * length / self._avg_len)
                score += self._idf.get(term, 0.0) * tf * (K1 + 1) / denom
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i].id,
                score=round(s, 4),
                text=self.chunks[i].text,
                section=self.chunks[i].section,
            )
            for s, i in scored[:top_k]
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None


def format_evidence(hits: list[RetrievedChunk], max_chars: int = 2400) -> str:
    """把检索结果渲染成给模型看的证据块,带 chunk 编号以便引用。"""
    out: list[str] = []
    used = 0
    for hit in hits:
        block = f"[{hit.chunk_id}] ({hit.section})\n{hit.text}"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)
