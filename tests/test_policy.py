"""策略层是纯函数,这里把教学上的关键性质都断言死。"""
import pytest

from tutor.config import PARAMS
from tutor.agent import policy
from tutor.schemas import (ActionType, ConceptState, Grade, LearningPath,
                           SessionState, Verdict)


def make_session(concept_ids=("k01", "k02")) -> SessionState:
    return SessionState(
        session_id="s", material_id="m",
        path=LearningPath(material_id="m", ordered_concept_ids=list(concept_ids)),
        concepts={c: ConceptState(concept_id=c, p_known=PARAMS.p_init) for c in concept_ids},
    )


def grade(score: float, tags=()) -> Grade:
    v = Verdict.CORRECT if score >= 0.8 else Verdict.PARTIAL if score >= 0.4 else Verdict.INCORRECT
    return Grade(score=score, verdict=v, misconception_tags=list(tags))


# --- BKT ---------------------------------------------------------------
def test_bkt_monotonic_in_score():
    p = 0.4
    assert policy.bkt_update(p, 0.0) < policy.bkt_update(p, 0.5) < policy.bkt_update(p, 1.0)


def test_bkt_stays_in_range():
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        for s in (0.0, 0.5, 1.0):
            assert 0.0 <= policy.bkt_update(p, s) <= 1.0


def test_persistent_half_credit_never_reaches_mastery():
    """关键性质:长期只拿一半分的学生不能被推到"掌握"。"""
    p = PARAMS.p_init
    for _ in range(50):
        p = policy.bkt_update(p, 0.5)
    assert p < PARAMS.mastery_threshold
    assert p < 0.6


def test_repeated_correct_converges_up():
    p = PARAMS.p_init
    for _ in range(4):
        p = policy.bkt_update(p, 1.0)
    assert p > PARAMS.mastery_threshold


def test_wrong_answers_pull_mastery_down():
    assert policy.bkt_update(0.9, 0.0) < 0.9


# --- 难度 --------------------------------------------------------------
def test_difficulty_tracks_mastery():
    low = policy.target_difficulty(ConceptState(concept_id="k", p_known=0.1), 5)
    high = policy.target_difficulty(ConceptState(concept_id="k", p_known=0.95), 5)
    assert low < high
    assert PARAMS.min_difficulty <= low and high <= PARAMS.max_difficulty


def test_difficulty_capped_by_concept_difficulty():
    """简单的知识点不该被出到 5 级难度。"""
    d = policy.target_difficulty(ConceptState(concept_id="k", p_known=0.99), concept_difficulty=1)
    assert d <= 2


# --- 达标判定 -----------------------------------------------------------
def test_high_probability_alone_is_not_mastery():
    """只靠两道简单题把概率顶上去,不算掌握。"""
    state = ConceptState(concept_id="k", p_known=0.95, attempts=2, max_difficulty_seen=1)
    assert not policy.is_mastered(state, concept_difficulty=4)


def test_mastery_requires_enough_attempts_and_difficulty():
    state = ConceptState(concept_id="k", p_known=0.95, attempts=3, max_difficulty_seen=3)
    assert policy.is_mastered(state, concept_difficulty=4)


def test_easy_concept_does_not_require_high_difficulty():
    state = ConceptState(concept_id="k", p_known=0.95, attempts=3, max_difficulty_seen=1)
    assert policy.is_mastered(state, concept_difficulty=1)


# --- 决策 --------------------------------------------------------------
def test_first_action_is_teach():
    s = make_session()
    assert policy.decide(s, {"k01": 3, "k02": 3}).action is ActionType.TEACH


def test_teach_then_practice():
    s = make_session()
    s.concepts["k01"].taught = True
    assert policy.decide(s, {"k01": 3, "k02": 3}).action is ActionType.PRACTICE


def test_repeated_misconception_triggers_remediation():
    s = make_session()
    st = s.concepts["k01"]
    st.taught = True
    st.p_known = 0.2
    st.misconceptions = {"k01-方向搞反": 2}
    plan = policy.decide(s, {"k01": 3, "k02": 3})
    assert plan.action is ActionType.REMEDIATE
    assert plan.focus_misconception == "k01-方向搞反"


def test_single_misconception_does_not_trigger_remediation():
    s = make_session()
    st = s.concepts["k01"]
    st.taught = True
    st.p_known = 0.2
    st.misconceptions = {"k01-偶发笔误": 1}
    assert policy.decide(s, {"k01": 3, "k02": 3}).action is ActionType.PRACTICE


def test_too_many_attempts_forces_advance():
    """防死循环:练满上限还没达标也要往前走,并留下复习债。"""
    s = make_session()
    st = s.concepts["k01"]
    st.taught = True
    st.attempts = PARAMS.max_items_per_concept
    plan = policy.decide(s, {"k01": 3, "k02": 3})
    assert plan.action is ActionType.ADVANCE
    policy.advance(s)
    assert any(r["concept_id"] == "k01" for r in s.review_queue)


# --- 复习 --------------------------------------------------------------
def test_review_fires_only_when_due():
    s = make_session()
    s.concepts["k01"].taught = True
    policy.schedule_review(s, "k01")
    assert policy.decide(s, {"k01": 3, "k02": 3}).action is not ActionType.REVIEW
    s.step += PARAMS.review_delay_steps
    assert policy.decide(s, {"k01": 3, "k02": 3}).action is ActionType.REVIEW


def test_failed_review_knocks_mastery_back():
    s = make_session()
    st = s.concepts["k01"]
    st.p_known = 0.95
    st.mastered_at_step = 0
    policy.apply_grade(s, "k01", grade(0.0), is_review=True)
    assert st.p_known < PARAMS.mastery_threshold
    assert st.mastered_at_step is None


def test_review_count_is_capped():
    """没有这个上限,「复习失败 → 打回 → 重新掌握 → 又排复习」会永远转下去。"""
    s = make_session()
    st = s.concepts["k01"]
    st.reviews = PARAMS.max_reviews_per_concept
    policy.schedule_review(s, "k01")
    assert s.review_queue == []


def test_review_is_not_scheduled_twice():
    s = make_session()
    policy.schedule_review(s, "k01")
    policy.schedule_review(s, "k01")
    assert len(s.review_queue) == 1


# --- 状态回写 -----------------------------------------------------------
def test_apply_grade_accumulates_misconceptions():
    s = make_session()
    policy.apply_grade(s, "k01", grade(0.0, ["k01-方向搞反"]), item_difficulty=2)
    policy.apply_grade(s, "k01", grade(0.0, ["k01-方向搞反"]), item_difficulty=2)
    assert s.concepts["k01"].misconceptions["k01-方向搞反"] == 2


def test_only_correct_answers_credit_difficulty():
    s = make_session()
    policy.apply_grade(s, "k01", grade(0.3), item_difficulty=5)
    assert s.concepts["k01"].max_difficulty_seen == 0
    policy.apply_grade(s, "k01", grade(1.0), item_difficulty=4)
    assert s.concepts["k01"].max_difficulty_seen == 4
