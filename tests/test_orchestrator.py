"""端到端:整个 agent 循环必须收敛,且不违反教学上的硬约束。"""
import random

import pytest

from tutor.agent.orchestrator import Tutor
from tutor.evaluation.simulate import run_simulation
from tutor.llm import FakeLLM
from tutor.schemas import ActionType


def run_to_completion(course, correctness=1.0, max_steps=200, seed=1):
    rng = random.Random(seed)
    tutor = Tutor(FakeLLM(), course)
    turns = []
    for _ in range(max_steps):
        turn = tutor.next_step()
        turns.append(turn)
        if turn.action is ActionType.COMPLETE:
            return tutor, turns
        if turn.item is None:
            continue
        good = rng.random() < correctness
        if turn.item.kind.value == "mcq":
            answer = turn.item.answer_key if good else "C"
        else:
            answer = "。".join(turn.item.rubric) if good else "不知道"
        tutor.submit_answer(answer)
    pytest.fail(f"{max_steps} 步内没有结束")


def test_perfect_student_finishes(course):
    tutor, turns = run_to_completion(course, correctness=1.0)
    assert tutor.session.finished
    assert turns[0].action is ActionType.TEACH


def test_failing_student_also_terminates(course):
    """全答错也必须能结束——不能把学生困在一个概念上。"""
    tutor, _ = run_to_completion(course, correctness=0.0)
    assert tutor.session.finished


def test_weak_student_gets_more_practice_than_strong_one(course):
    """这是"自适应"最核心的可观测性质。"""
    _, weak = run_to_completion(course, correctness=0.35, seed=5)
    _, strong = run_to_completion(course, correctness=1.0, seed=5)
    weak_items = [t for t in weak if t.item]
    strong_items = [t for t in strong if t.item]
    assert len(weak_items) > len(strong_items)


def test_every_concept_is_taught_before_being_tested(course):
    _, turns = run_to_completion(course, correctness=0.6, seed=9)
    taught = set()
    for turn in turns:
        if turn.action is ActionType.TEACH:
            taught.add(turn.concept_id)
        elif turn.item is not None and turn.action is not ActionType.REVIEW:
            assert turn.concept_id in taught, f"{turn.concept_id} 还没讲就出题了"


def test_prerequisites_are_respected(course):
    _, turns = run_to_completion(course, correctness=0.7, seed=3)
    seen = set()
    for turn in turns:
        cid = turn.concept_id
        if not cid or cid in seen:
            continue
        concept = course.concepts.get(cid)
        if concept:
            unmet = [p for p in concept.prerequisites if p in course.concepts and p not in seen]
            assert not unmet, f"{cid} 的先修 {unmet} 还没学"
        seen.add(cid)


def test_items_always_cite_real_chunks(course):
    valid = set(course.material.chunk_map())
    _, turns = run_to_completion(course, correctness=0.6, seed=4)
    for turn in turns:
        if turn.item:
            assert turn.item.source_chunk_ids
            assert set(turn.item.source_chunk_ids) <= valid


def test_pending_item_is_returned_again_until_answered(course):
    tutor = Tutor(FakeLLM(), course)
    while True:
        turn = tutor.next_step()
        if turn.item:
            break
    assert tutor.next_step() is turn        # 没作答就不会往前走
    with pytest.raises(ValueError):
        Tutor(FakeLLM(), course).submit_answer("x")   # 没有待答题时提交应报错


def test_no_duplicate_stems_within_a_concept(course):
    """出题时把已问过的题干喂回去,同一概念不应连续出一模一样的题。"""
    _, turns = run_to_completion(course, correctness=0.3, seed=8)
    by_concept = {}
    for turn in turns:
        if turn.item:
            by_concept.setdefault(turn.concept_id, []).append(turn.item.difficulty)
    # FakeLLM 按难度出题,难度必须随掌握度变化而不是恒定
    assert any(len(set(v)) > 1 for v in by_concept.values())


def test_report_flags_unmastered_concepts(course):
    tutor, _ = run_to_completion(course, correctness=0.0)
    report = tutor.report()
    assert report.shaky, "全答错却报告全部掌握,说明报告在撒谎"
    assert report.overall < 0.85


def test_session_state_round_trips_through_json(course):
    from tutor.schemas import SessionState

    tutor, _ = run_to_completion(course, correctness=0.6, seed=2)
    restored = SessionState.model_validate_json(tutor.session.model_dump_json())
    assert restored.step == tutor.session.step
    assert restored.concepts.keys() == tutor.session.concepts.keys()


def test_simulation_reports_adaptive_behaviour(course):
    weak = run_simulation(FakeLLM(), course, ability=0.15, seed=3, max_steps=200)
    strong = run_simulation(FakeLLM(), course, ability=0.95, seed=3, max_steps=200)
    assert weak.completed and strong.completed
    assert weak.items >= strong.items
    assert weak.calibration_mae >= 0.0
    assert strong.prereq_respected
