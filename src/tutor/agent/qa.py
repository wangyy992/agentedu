"""智能答疑:学生随时可以打断提问。

这是唯一一个"模型自主决定行动"的环节——它自己决定搜什么、搜几次、
要不要查学生的掌握状态,然后给出带引用的回答。回答后会做一次
**引用核验**:模型声称的 [c0xx] 如果不在真实检索结果里,就降级标记为未溯源。
"""
from __future__ import annotations

import re

from ..llm import LLMClient, Prompt
from ..retrieval.bm25 import Retriever
from ..schemas import GroundedAnswer, Material, SessionState
from .tools import TOOLS, build_executor

SYSTEM = """你在回答学生关于学习材料的提问。规则:
1. 回答前**必须**先用 search_material 找依据;一次没找到就换关键词再找一次。
2. 每个论断后标注来源片段编号 [c0xx];没有依据的话不要说。
3. 材料确实没写的,直接说「材料里没有讲到这一点」,可以补一句它属于什么范畴,
   但要明确标注这是材料之外的补充。
4. 如果问题涉及学生自己的掌握情况或"该复习什么",调用 get_student_progress。
5. 回答控制在 250 字以内,先给结论再给依据。不要出题。"""


def answer_question(
    llm: LLMClient,
    retriever: Retriever,
    material: Material,
    question: str,
    *,
    session: SessionState | None = None,
    concept_names: dict[str, str] | None = None,
    cached_context: str = "",
) -> GroundedAnswer:
    executor = build_executor(retriever, session, concept_names)
    prompt = Prompt(
        task="answer_question",
        system=SYSTEM,
        user=f"学生的问题:{question}",
        data={"question": question},
    )
    result = llm.tool_loop(
        prompt,
        tools=TOOLS,
        executor=executor,
        cached_context=cached_context,
        max_turns=5,
    )

    # 核验引用:只认真实出现在工具返回里的片段编号
    retrieved: set[str] = set()
    for call in result.tool_calls:
        retrieved.update(re.findall(r"\[(c\d+)\]", call.result))
    claimed = set(re.findall(r"\[(c\d+)\]", result.text))
    verified = sorted(claimed & retrieved)
    hallucinated = sorted(claimed - retrieved)

    text = result.text
    if hallucinated:
        text += f"\n\n(提示:引用 {', '.join(hallucinated)} 未能在材料中核实,请以标注为准。)"

    return GroundedAnswer(
        answer=text,
        citations=verified,
        used_searches=[str(c.arguments.get("query", "")) for c in result.tool_calls
                       if c.name == "search_material"],
        grounded=bool(verified),
    )
