"""第 2 步:规划学习路径。

拓扑排序是**纯代码**做的,不交给模型——顺序正确性是硬约束,
不该受模型抖动影响。模型只负责在图里给出依赖关系(上一步),
这里负责把图变成一条稳定、可解释、且能处理环的执行序列。
"""
from __future__ import annotations

from ..schemas import ConceptGraph, LearningPath


def plan_path(graph: ConceptGraph) -> LearningPath:
    concepts = {c.id: c for c in graph.concepts}
    original_order = {cid: i for i, cid in enumerate(concepts)}

    # 入度表(只统计图内真实存在的先修)
    prereqs = {cid: {p for p in c.prerequisites if p in concepts} for cid, c in concepts.items()}
    indeg = {cid: len(ps) for cid, ps in prereqs.items()}

    ordered: list[str] = []
    cycles_broken: list[str] = []
    remaining = set(concepts)

    while remaining:
        ready = [cid for cid in remaining if indeg[cid] == 0]
        if not ready:
            # 存在环:选一个"先修最少、在原文中最靠前"的强行放行,并记录下来
            victim = min(
                remaining, key=lambda c: (indeg[c], concepts[c].difficulty, original_order[c])
            )
            cycles_broken.append(victim)
            indeg[victim] = 0
            ready = [victim]

        # 同层内:先易后难,难度相同则按原文顺序 —— 保证结果确定可复现
        ready.sort(key=lambda c: (concepts[c].difficulty, original_order[c]))
        chosen = ready[0]
        ordered.append(chosen)
        remaining.discard(chosen)
        for cid in remaining:
            if chosen in prereqs[cid]:
                prereqs[cid].discard(chosen)
                indeg[cid] -= 1

    return LearningPath(
        material_id=graph.material_id,
        ordered_concept_ids=ordered,
        rationale=_explain(concepts, ordered, cycles_broken),
        cycles_broken=cycles_broken,
    )


def _explain(concepts: dict, ordered: list[str], cycles: list[str]) -> str:
    head = "、".join(concepts[c].name for c in ordered[:3])
    tail = "、".join(concepts[c].name for c in ordered[-2:]) if len(ordered) > 3 else ""
    text = f"按先修依赖拓扑排序,同层由易到难。先学:{head}"
    if tail:
        text += f";最后收尾:{tail}"
    if cycles:
        names = "、".join(concepts[c].name for c in cycles)
        text += f"。注意:{names} 存在循环依赖,已按难度最低者优先切开。"
    return text
