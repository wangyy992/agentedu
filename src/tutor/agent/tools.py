"""给模型用的工具集(答疑场景)。

这里是项目里唯一一处**模型自己决定调用什么、调用几次**的地方:
学生提问 -> 模型自行检索材料(可多轮、可换关键词)-> 给出带引用的回答。
工具定义用 strict schema,保证参数一定合法。
"""
from __future__ import annotations

import json
from typing import Any, Callable

from ..retrieval.bm25 import Retriever
from ..schemas import SessionState

SEARCH_MATERIAL = {
    "name": "search_material",
    "description": (
        "在当前学习材料中做关键词检索,返回最相关的片段及其编号。"
        "回答任何与材料内容有关的问题前都应先调用它取得依据;"
        "第一次没搜到就换关键词再搜一次。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词,用材料里可能出现的术语"},
            "top_k": {"type": "integer", "description": "返回片段数,1~8"},
        },
        "required": ["query", "top_k"],
        "additionalProperties": False,
    },
}

GET_PROGRESS = {
    "name": "get_student_progress",
    "description": (
        "查询这位学生当前的学习状态:各知识点的掌握度、做过多少题、有哪些反复出现的误区。"
        "当问题涉及「我学得怎么样」「我该复习什么」或需要因材施教时调用。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "concept_id": {
                "type": "string",
                "description": "只看某个知识点就填它的 id;看全部填空字符串",
            }
        },
        "required": ["concept_id"],
        "additionalProperties": False,
    },
}

TOOLS = [SEARCH_MATERIAL, GET_PROGRESS]


def build_executor(
    retriever: Retriever,
    session: SessionState | None = None,
    concept_names: dict[str, str] | None = None,
) -> Callable[[str, dict[str, Any]], str]:
    """返回一个 (工具名, 参数) -> 字符串结果 的分发函数。"""
    names = concept_names or {}

    def execute(name: str, args: dict[str, Any]) -> str:
        if name == "search_material":
            top_k = int(args.get("top_k") or 5)
            hits = retriever.search(str(args.get("query", "")), top_k=max(1, min(8, top_k)))
            if not hits:
                return "未检索到相关片段。请换一个关键词重试,或告诉学生材料未涉及该内容。"
            return "\n\n".join(
                f"[{h.chunk_id}] ({h.section}) score={h.score}\n{h.text}" for h in hits
            )

        if name == "get_student_progress":
            if session is None:
                return "当前没有进行中的学习会话。"
            cid = (args.get("concept_id") or "").strip()
            states = (
                [session.concepts[cid]] if cid and cid in session.concepts
                else list(session.concepts.values())
            )
            if not states:
                return "该学生尚无作答记录。"
            payload = [
                {
                    "concept": names.get(s.concept_id, s.concept_id),
                    "掌握度": round(s.p_known, 2),
                    "答题数": s.attempts,
                    "答对数": s.correct,
                    "反复误区": sorted(s.misconceptions, key=s.misconceptions.get, reverse=True)[:3],
                }
                for s in states
            ]
            return json.dumps(payload, ensure_ascii=False, indent=2)

        raise ValueError(f"未知工具: {name}")

    return execute
