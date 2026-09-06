"""第 3 步:讲解。

讲解不是"复述材料",而是按学生**当前状态**调整:
- 首次讲解 -> 完整讲一遍;
- 补救讲解 -> 只针对反复出错的那个误区,换个角度重讲,并给出反例。
证据由检索给出,要求逐点标注 chunk 编号——这是可溯源的前提。
"""
from __future__ import annotations

from ..config import PARAMS
from ..llm import LLMClient, Prompt
from ..retrieval.bm25 import Retriever, format_evidence
from ..schemas import Concept, ConceptState

TEACH_SYSTEM = """你在给一位初学者讲解一个知识点。要求:
1. 200~400 字,结构为:一句话定义 -> 为什么需要它 -> 一个具体例子 -> 一句话小结。
2. 只使用 <证据> 中的信息;每个关键论断后用 [c0xx] 标注来源片段。
3. 语言直白,不堆术语;出现术语必须当场解释。
4. 不要出题,不要说"接下来我们做几道题"。"""

REMEDIATE_SYSTEM = """学生在这个知识点上反复出现同一个误区。请做针对性补救:
1. 先**直接点出**这个误解错在哪(不要绕弯子)。
2. 用一个和之前不同的角度重新解释(换类比、换表述、或给一个反例)。
3. 给一条可操作的判别方法,让他下次能自己识别。
4. 150~250 字,关键论断标注 [c0xx]。不要出题。"""


def explain(
    llm: LLMClient,
    retriever: Retriever,
    concept: Concept,
    *,
    cached_context: str = "",
) -> tuple[str, list[str]]:
    hits = retriever.search(f"{concept.name} {concept.summary}", top_k=PARAMS.retrieve_top_k)
    evidence = format_evidence(hits)
    prompt = Prompt(
        task="explain",
        system=TEACH_SYSTEM,
        user=(
            f"知识点:{concept.name}\n"
            f"摘要:{concept.summary}\n"
            f"要点:{'; '.join(concept.key_points)}\n\n"
            f"<证据>\n{evidence}\n</证据>\n\n请讲解这个知识点。"
        ),
        data={
            "concept_name": concept.name,
            "key_points": concept.key_points,
            "evidence": evidence,
        },
    )
    return llm.text(prompt, cached_context=cached_context), [h.chunk_id for h in hits]


def remediate(
    llm: LLMClient,
    retriever: Retriever,
    concept: Concept,
    state: ConceptState,
    misconception: str,
    *,
    cached_context: str = "",
) -> tuple[str, list[str]]:
    # 检索时把误区本身也当查询词,才能捞到真正相关的反驳材料
    hits = retriever.search(f"{concept.name} {misconception}", top_k=PARAMS.retrieve_top_k)
    evidence = format_evidence(hits)
    prompt = Prompt(
        task="explain",
        system=REMEDIATE_SYSTEM,
        user=(
            f"知识点:{concept.name}\n"
            f"学生的误区:{misconception}(已出现 {state.misconceptions.get(misconception, 1)} 次)\n"
            f"当前掌握度估计:{state.p_known:.2f}\n\n"
            f"<证据>\n{evidence}\n</证据>\n\n请做针对性补救讲解。"
        ),
        data={
            "concept_name": f"{concept.name}(补救:{misconception})",
            "key_points": concept.key_points,
            "evidence": evidence,
        },
    )
    return llm.text(prompt, cached_context=cached_context), [h.chunk_id for h in hits]
