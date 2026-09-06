"""第 4 步:出题。

三个硬约束:
1. **可溯源** —— 每道题必须绑定 source_chunk_ids,答案能在材料里找到依据;
2. **难度可控** —— 难度 1~5 对应 Bloom 认知层级,由策略层指定而非模型自由发挥;
3. **不重复** —— 把本概念已出过的题干喂回去,要求换角度。
"""
from __future__ import annotations

import logging

from ..config import PARAMS
from ..llm import LLMClient, Prompt
from ..retrieval.bm25 import Retriever, format_evidence
from ..schemas import Concept, Item, ItemKind

log = logging.getLogger("tutor.item_writer")

BLOOM = {
    1: "记忆:直接回忆定义或事实",
    2: "理解:用自己的话复述、判断说法对错",
    3: "应用:在一个新的具体情境里使用它",
    4: "分析:比较、拆解、找出因果或反例",
    5: "综合:结合多个知识点做判断或设计",
}

SYSTEM = """你在为一位学生出**一道**练习题。要求:
1. 严格按指定的难度层级出题(见用户消息中的 Bloom 描述)。
2. 题目答案必须能在 <证据> 里找到依据;source_chunk_ids 填实际依据的片段编号。
3. kind 为 mcq 时:options 恰好 4 项,格式 "A. xxx",answer_key 只填字母;
   干扰项要是**常见误解**,不能是明显的胡话。
   kind 为 short/cloze 时:options 留空数组,answer_key 填参考答案。
4. rubric 写 2~4 条可逐条核对的评分要点(简答题判分完全依赖它)。
5. rationale 说明为什么该答案正确;hint 是答错后的第一级提示,只点方向不给答案。
6. 不要出现"以上都对/都不对"这类懒惰选项。只输出 JSON。"""


def write_item(
    llm: LLMClient,
    retriever: Retriever,
    concept: Concept,
    difficulty: int,
    *,
    asked_stems: list[str] | None = None,
    focus_misconception: str | None = None,
    cached_context: str = "",
) -> Item:
    difficulty = max(PARAMS.min_difficulty, min(PARAMS.max_difficulty, difficulty))
    query = f"{concept.name} {concept.summary}"
    if focus_misconception:
        query += f" {focus_misconception}"
    hits = retriever.search(query, top_k=PARAMS.retrieve_top_k)
    evidence = format_evidence(hits)

    avoid = ""
    if asked_stems:
        avoid = "\n已经问过的题(换个角度,不要重复):\n" + "\n".join(f"- {s}" for s in asked_stems[-4:])
    focus = f"\n本题要专门检验学生是否还存在这个误区:{focus_misconception}" if focus_misconception else ""

    prompt = Prompt(
        task="write_item",
        system=SYSTEM,
        user=(
            f"知识点:{concept.name}\n摘要:{concept.summary}\n"
            f"难度:{difficulty} —— {BLOOM[difficulty]}{focus}{avoid}\n\n"
            f"<证据>\n{evidence}\n</证据>\n\n请出一道题。"
        ),
        data={
            "concept_name": concept.name,
            "key_points": concept.key_points,
            "difficulty": difficulty,
            "chunk_ids": [h.chunk_id for h in hits],
        },
    )
    item = llm.structured(prompt, Item, cached_context=cached_context)
    return _normalize(item, concept, difficulty, [h.chunk_id for h in hits])


def _normalize(item: Item, concept: Concept, difficulty: int, valid_chunks: list[str]) -> Item:
    """把模型的输出收进可信区间——出题环节最容易出格式问题,这里统一兜住。"""
    item.concept_id = concept.id
    item.difficulty = difficulty
    item.source_chunk_ids = [c for c in item.source_chunk_ids if c in valid_chunks] or valid_chunks[:2]

    if item.kind == ItemKind.MCQ:
        if len(item.options) < 2:
            log.warning("选择题选项不足,降级为简答题:%s", item.stem[:40])
            item.kind = ItemKind.SHORT
            item.options = []
        else:
            key = item.answer_key.strip().upper()[:1]
            letters = [chr(ord("A") + i) for i in range(len(item.options))]
            if key not in letters:
                # 模型有时把答案原文而非字母填进 answer_key,尝试对回去
                matched = next(
                    (l for l, o in zip(letters, item.options) if item.answer_key.strip() in o), None
                )
                item.answer_key = matched or letters[0]
            else:
                item.answer_key = key
    if not item.rubric:
        item.rubric = [f"正确说明「{concept.name}」的核心含义"]
    return item
