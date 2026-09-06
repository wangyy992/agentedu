"""第 1 步:理解材料 —— 抽取知识点及其先修关系。

这是整条链路里唯一一次"读全文"的调用,因此把材料放进 prompt cache,
后面几十次出题/讲解的调用都能复用这段前缀。
"""
from __future__ import annotations

import logging

from ..llm import LLMClient, Prompt
from ..schemas import ConceptGraph, Material

log = logging.getLogger("tutor.comprehend")

SYSTEM = """你在为一份学习材料建立知识图谱。要求:
1. 抽出 5~15 个**可教可考**的知识点(过细的例子、无信息量的过渡段不算)。
2. 每个知识点给出 id(k01、k02……按材料出现顺序)、名称、一句话摘要、2~4 条要点。
3. prerequisites 只能填**本材料内**其它知识点的 id。没有先修就留空数组。
   只在真正存在依赖时才连边(不懂 A 就学不会 B),不要为了连而连。
4. evidence_chunk_ids 必须是材料中真实出现过的编号(形如 c007),用于后续出题溯源。
5. difficulty 取 1~5,表示该知识点本身的认知难度。
只输出 JSON。"""


def extract_concepts(llm: LLMClient, material: Material) -> ConceptGraph:
    sections = _section_digest(material)
    user = (
        f"材料标题:{material.title}\n"
        f"共 {len(material.chunks)} 个片段,小节结构如下:\n"
        + "\n".join(f"- {s['title']}(片段 {', '.join(s['chunk_ids'])})" for s in sections)
        + "\n\n请基于上文 <学习材料> 全文建立知识图谱。"
    )
    prompt = Prompt(
        task="extract_concepts",
        system=SYSTEM,
        user=user,
        data={"material_id": material.id, "sections": sections},
    )
    graph = llm.structured(prompt, ConceptGraph, cached_context=material.full_text())
    graph.material_id = material.id
    return _sanitize(graph, material)


def _section_digest(material: Material) -> list[dict]:
    """按小节聚合 chunk,给模型一个结构骨架(也是 FakeLLM 的输入)。"""
    buckets: dict[str, dict] = {}
    for chunk in material.chunks:
        key = chunk.section or material.title
        bucket = buckets.setdefault(key, {"title": key, "text": "", "chunk_ids": []})
        bucket["chunk_ids"].append(chunk.id)
        if len(bucket["text"]) < 500:
            bucket["text"] += chunk.text[:300] + " "
    return list(buckets.values())


def _sanitize(graph: ConceptGraph, material: Material) -> ConceptGraph:
    """信任但要核验:剔除模型编造的 chunk 编号和指向不存在概念的先修边。"""
    valid_chunks = set(material.chunk_map())
    valid_concepts = {c.id for c in graph.concepts}
    for concept in graph.concepts:
        bad_chunks = [x for x in concept.evidence_chunk_ids if x not in valid_chunks]
        if bad_chunks:
            log.warning("概念 %s 引用了不存在的片段 %s,已剔除", concept.id, bad_chunks)
        concept.evidence_chunk_ids = [x for x in concept.evidence_chunk_ids if x in valid_chunks]
        concept.prerequisites = [
            p for p in concept.prerequisites if p in valid_concepts and p != concept.id
        ]
        concept.difficulty = max(1, min(5, concept.difficulty))
    return graph
