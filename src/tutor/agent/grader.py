"""第 5 步:判分。

混合策略,不是"什么都丢给模型":
- 选择题选对 -> 纯代码判定满分,直接复用出题时生成的 rationale,**零 token**;
- 选择题选错 / 简答题 -> 交给模型按 rubric 逐条核对,并产出**误区标签**。

误区标签是整套自适应的燃料:它决定补救讲什么、下一题考什么。
所以 prompt 里要求标签规范化(知识点-误解类型),否则标签会碎成一次性字符串,
无法在多次作答之间累计。
"""
from __future__ import annotations

from ..llm import LLMClient, Prompt
from ..retrieval.bm25 import Retriever, format_evidence
from ..schemas import Grade, Item, ItemKind, Verdict

SYSTEM = """你在批改一道练习题。要求:
1. 逐条核对 rubric:命中的放进 matched_rubric,没命中的放进 missing_points。
   score = 命中条数 / 总条数,可给部分分。
2. verdict:score ≥ 0.8 为 correct,0.4~0.8 为 partial,低于 0.4 为 incorrect。
3. feedback 写给学生看:先肯定对的部分,再**具体**指出错在哪(不要"再想想"这种废话),
   最后给一句能立刻用上的提示。80~150 字。
4. misconception_tags 是关键:用「知识点-误解类型」的规范格式,例如
   "学习率-把大学习率当成收敛快"、"梯度下降-混淆梯度方向"。
   同一种误解在不同题目中必须产出**同一个标签**,以便跨题累计。
   答对则留空数组。最多 2 个标签。
5. 只依据 <证据> 判分;学生说的即使听起来合理,材料里没有依据就不算得分点。
只输出 JSON。"""


def grade(
    llm: LLMClient,
    retriever: Retriever,
    item: Item,
    answer: str,
    concept_name: str,
    *,
    cached_context: str = "",
) -> Grade:
    answer = (answer or "").strip()
    if not answer:
        return Grade(
            score=0.0,
            verdict=Verdict.INCORRECT,
            feedback="没有作答。先回到讲解部分,再试一次——不确定也可以写下你的思路。",
            missing_points=list(item.rubric),
            misconception_tags=[],
        )

    # 快路径:选择题答对,规则即可判定,省一次模型调用
    if item.kind == ItemKind.MCQ:
        picked = _parse_choice(answer, len(item.options))
        if picked and picked == item.answer_key:
            return Grade(
                score=1.0,
                verdict=Verdict.CORRECT,
                feedback=f"正确。{item.rationale}",
                matched_rubric=list(item.rubric),
            )

    hits = retriever.search(f"{concept_name} {item.stem}", top_k=4)
    evidence = format_evidence(hits, max_chars=1600)
    options = "\n".join(item.options) if item.options else "(简答题)"

    prompt = Prompt(
        task="grade",
        system=SYSTEM,
        user=(
            f"知识点:{concept_name}\n题型:{item.kind.value}\n难度:{item.difficulty}\n"
            f"题干:{item.stem}\n选项:\n{options}\n"
            f"参考答案:{item.answer_key}\n"
            f"评分要点:\n" + "\n".join(f"- {r}" for r in item.rubric) + "\n\n"
            f"学生的回答:\n{answer}\n\n"
            f"<证据>\n{evidence}\n</证据>\n\n请判分。"
        ),
        data={"answer": answer, "rubric": item.rubric, "concept_name": concept_name},
    )
    result = llm.structured(prompt, Grade, cached_context=cached_context)
    return _reconcile(result, item, answer)


def _parse_choice(answer: str, n_options: int) -> str | None:
    """从"我选B"、"b)"、"B. xxx" 之类的自由输入里抠出选项字母。"""
    letters = {chr(ord("A") + i) for i in range(max(0, n_options))}
    for ch in answer.upper():
        if ch in letters:
            return ch
    return None


def _reconcile(result: Grade, item: Item, answer: str) -> Grade:
    """对齐 score 与 verdict,防止模型给出自相矛盾的组合。"""
    if item.kind == ItemKind.MCQ:
        picked = _parse_choice(answer, len(item.options))
        if picked and picked != item.answer_key:
            # 选择题选错就是错,不给部分分——避免模型对"思路不错"心软
            result.score = min(result.score, 0.2)
    result.score = max(0.0, min(1.0, result.score))
    result.verdict = (
        Verdict.CORRECT if result.score >= 0.8
        else Verdict.PARTIAL if result.score >= 0.4
        else Verdict.INCORRECT
    )
    result.misconception_tags = [t.strip() for t in result.misconception_tags if t.strip()][:2]
    if result.verdict == Verdict.CORRECT:
        result.misconception_tags = []
    return result
