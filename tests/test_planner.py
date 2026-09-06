from tutor.agent.planner import plan_path
from tutor.schemas import Concept, ConceptGraph


def graph(*specs) -> ConceptGraph:
    return ConceptGraph(
        material_id="m",
        concepts=[Concept(id=i, name=i, prerequisites=list(p), difficulty=d) for i, p, d in specs],
    )


def test_prerequisites_come_first():
    order = plan_path(graph(("c", ["b"], 3), ("a", [], 1), ("b", ["a"], 2))).ordered_concept_ids
    assert order.index("a") < order.index("b") < order.index("c")


def test_same_layer_sorted_easy_first():
    order = plan_path(graph(("hard", [], 5), ("easy", [], 1))).ordered_concept_ids
    assert order == ["easy", "hard"]


def test_cycle_is_broken_and_reported():
    path = plan_path(graph(("a", ["b"], 3), ("b", ["a"], 1)))
    assert set(path.ordered_concept_ids) == {"a", "b"}
    assert path.cycles_broken == ["b"]           # 难度低的先放行
    assert "循环依赖" in path.rationale


def test_dangling_prerequisite_is_ignored():
    """指向不存在概念的先修边不能把节点永久锁死。"""
    path = plan_path(graph(("a", ["ghost"], 2),))
    assert path.ordered_concept_ids == ["a"]


def test_every_concept_appears_exactly_once():
    g = graph(*[(f"k{i}", [f"k{i-1}"] if i else [], i % 5 + 1) for i in range(12)])
    order = plan_path(g).ordered_concept_ids
    assert len(order) == 12 and len(set(order)) == 12


def test_planning_is_deterministic():
    g = graph(("a", [], 2), ("b", [], 2), ("c", [], 2))
    assert plan_path(g).ordered_concept_ids == plan_path(g).ordered_concept_ids
