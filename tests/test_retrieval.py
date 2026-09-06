from tutor.ingest.chunker import chunk_material
from tutor.retrieval.bm25 import BM25Retriever, format_evidence, tokenize


def test_tokenize_handles_chinese_and_english():
    tokens = tokenize("Adam 优化器")
    assert "adam" in tokens
    assert "优化" in tokens      # bigram
    assert "器" in tokens        # unigram


def test_search_finds_the_right_section(sample_text):
    r = BM25Retriever(chunk_material("m", "t", sample_text).chunks)
    top = r.search("学习率过大会怎样", top_k=1)[0]
    assert "学习率" in top.text


def test_search_is_ranked_and_bounded(sample_text):
    r = BM25Retriever(chunk_material("m", "t", sample_text).chunks)
    hits = r.search("动量 自适应 Adam", top_k=3)
    assert len(hits) <= 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_empty_query_returns_nothing(sample_text):
    r = BM25Retriever(chunk_material("m", "t", sample_text).chunks)
    assert r.search("") == []


def test_format_evidence_respects_budget(sample_text):
    r = BM25Retriever(chunk_material("m", "t", sample_text).chunks)
    text = format_evidence(r.search("梯度", top_k=8), max_chars=300)
    assert len(text) <= 400 and "[c" in text
