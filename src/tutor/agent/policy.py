"""自适应策略层:决定"下一步教什么、考什么、多难"。

这一层**完全不调用 LLM**,是纯函数 + 显式状态机。这样做的好处:
- 可单测、可复现、可做消融实验;
- 模型抖动不会让教学路径乱跳;
- 出问题时能精确说明"为什么给你出这道题"(reason 字段一路带到前端)。

掌握度用贝叶斯知识追踪(BKT)建模:
    P(答对)   = L·(1-slip) + (1-L)·guess
    答对后    L' = L(1-slip) / [L(1-slip) + (1-L)·guess]
    答错后    L' = L·slip   / [L·slip   + (1-L)(1-guess)]
    再学习     L'' = L' + (1-L')·transit
标准 BKT 只吃 0/1,这里的判分是 [0,1] 连续分,所以在两个后验之间按分数插值。
"""
from __future__ import annotations

from ..config import PARAMS, TutorParams
from ..schemas import (
    ActionType,
    ConceptState,
    Grade,
    LearningPath,
    SessionState,
    StepPlan,
)


# --- BKT ----------------------------------------------------------------
def bkt_update(p_known: float, score: float, params: TutorParams = PARAMS) -> float:
    """用一次得分更新掌握概率。score ∈ [0,1],支持部分给分。"""
    p_known = _clamp(p_known, 1e-4, 1 - 1e-4)
    score = _clamp(score, 0.0, 1.0)

    correct_num = p_known * (1 - params.p_slip)
    correct_den = correct_num + (1 - p_known) * params.p_guess
    post_correct = correct_num / correct_den if correct_den > 0 else p_known

    wrong_num = p_known * params.p_slip
    wrong_den = wrong_num + (1 - p_known) * (1 - params.p_guess)
    post_wrong = wrong_num / wrong_den if wrong_den > 0 else p_known

    # 连续分数 => 在"全对后验"和"全错后验"之间插值
    posterior = post_wrong + (post_correct - post_wrong) * score

    # 学习效应:做题本身也在教学生。但**必须只在明确答对时才给增益**——
    # 若像常见写法那样给一个恒正的增益,一个长期只拿半分的学生会被这一项
    # 慢慢推过掌握阈值(评测里的校准误差就是这么来的)。
    # 现在半分及以下 learn=0,纯贝叶斯更新会收敛到 ~0.46 的不动点,永远不会误判掌握。
    learn = params.p_transit * max(0.0, (score - 0.5) * 2)
    return _clamp(posterior + (1 - posterior) * learn, 0.0, 0.999)


def p_correct(p_known: float, params: TutorParams = PARAMS) -> float:
    """当前状态下预测答对的概率——用来挑「不太难也不太简单」的题。"""
    return p_known * (1 - params.p_slip) + (1 - p_known) * params.p_guess


# --- 难度选择 -----------------------------------------------------------
def target_difficulty(
    state: ConceptState, concept_difficulty: int, params: TutorParams = PARAMS
) -> int:
    """把掌握度映射到 1~5 的难度。

    对准「预期答对率 ~0.7」这个甜区:太简单学不到东西,太难打击信心。
    concept_difficulty 是概念本身的固有难度,作为上限约束,避免给简单概念出压轴题。
    """
    base = 1 + round(4 * _clamp(state.p_known, 0.0, 1.0))
    # 连续答对/答错的即时微调
    if state.attempts >= 2:
        rate = state.correct / state.attempts
        if rate >= 0.8:
            base += 1
        elif rate <= 0.3:
            base -= 1
    ceiling = min(params.max_difficulty, concept_difficulty + 1)
    return int(_clamp(base, params.min_difficulty, ceiling))


# --- 达标判定 -----------------------------------------------------------
def is_mastered(state: ConceptState, concept_difficulty: int,
                params: TutorParams = PARAMS) -> bool:
    """三个条件同时满足才算掌握。

    只看 BKT 概率是不够的:连对两道难度 1 的题也能把 p_known 顶到 0.9,
    但那只证明学生记住了定义。所以再加两条**证据强度**约束——
    题量够、且在足够高的认知层级上答对过。concept_difficulty 是上限,
    本身就很简单的知识点不强求 4、5 级难度。
    """
    if state.p_known < params.mastery_threshold:
        return False
    if state.attempts < params.min_attempts_for_mastery:
        return False
    required = min(params.min_difficulty_for_mastery, max(1, concept_difficulty))
    return state.max_difficulty_seen >= required


# --- 决策 ---------------------------------------------------------------
def decide(session: SessionState, concept_difficulties: dict[str, int],
           params: TutorParams = PARAMS) -> StepPlan:
    """状态机的核心:看当前状态,给出下一步动作。

    优先级:到期复习 > 当前概念未讲 > 需要补救 > 达标推进 > 继续练习。
    """
    due = _due_review(session)
    if due:
        st = session.concepts[due]
        gap = session.step - st.mastered_at_step if st.mastered_at_step is not None else session.step
        return StepPlan(
            action=ActionType.REVIEW,
            concept_id=due,
            difficulty=max(params.min_difficulty, st.difficulty - 1),
            reason=f"「{due}」距上次掌握已过 {gap} 步,做第 {st.reviews + 1} 次间隔复习防遗忘",
        )

    concept_id = session.current_concept_id()
    if concept_id is None:
        return StepPlan(action=ActionType.COMPLETE, reason="学习路径已全部走完")

    state = session.concepts.setdefault(concept_id, ConceptState(concept_id=concept_id))
    concept_difficulty = concept_difficulties.get(concept_id, 3)

    if not state.taught:
        return StepPlan(
            action=ActionType.TEACH,
            concept_id=concept_id,
            difficulty=target_difficulty(state, concept_difficulty, params),
            reason="首次接触该概念,先讲解",
        )

    # 达标:推进
    if is_mastered(state, concept_difficulty, params):
        return StepPlan(
            action=ActionType.ADVANCE,
            concept_id=concept_id,
            reason=(f"掌握度 {state.p_known:.2f} ≥ {params.mastery_threshold},"
                    f"且已在难度 {state.max_difficulty_seen} 上答对过,进入下一个概念"),
        )

    # 练太多次还没上去:标记为待巩固,先往前走,靠复习队列回收
    if state.attempts >= params.max_items_per_concept:
        return StepPlan(
            action=ActionType.ADVANCE,
            concept_id=concept_id,
            reason=f"已练 {state.attempts} 题仍未达标,先推进并放入复习队列,避免卡死",
        )

    # 有反复出现的误区:针对性补救
    focus = _dominant_misconception(state)
    if focus and state.p_known < params.remediate_threshold:
        return StepPlan(
            action=ActionType.REMEDIATE,
            concept_id=concept_id,
            difficulty=max(params.min_difficulty, state.difficulty - 1),
            reason=f"掌握度偏低且反复出现「{focus}」,做针对性补救",
            focus_misconception=focus,
        )

    return StepPlan(
        action=ActionType.PRACTICE,
        concept_id=concept_id,
        difficulty=target_difficulty(state, concept_difficulty, params),
        reason=f"掌握度 {state.p_known:.2f},出一道难度 "
               f"{target_difficulty(state, concept_difficulty, params)} 的题继续练",
    )


def apply_grade(session: SessionState, concept_id: str, grade: Grade,
                is_review: bool = False, params: TutorParams = PARAMS,
                item_difficulty: int = 0, concept_difficulty: int = 3) -> ConceptState:
    """把判分结果写回状态:更新 BKT、误区计数、难度、复习队列。"""
    state = session.concepts.setdefault(concept_id, ConceptState(concept_id=concept_id))
    state.attempts += 1
    if grade.verdict.value == "correct":
        state.correct += 1
    state.last_step = session.step
    if grade.score >= 0.8:
        # 只有答对才算"在这个难度上有过证据"
        state.max_difficulty_seen = max(state.max_difficulty_seen, item_difficulty)

    if is_review:
        state.reviews += 1

    if is_review and grade.score < 0.6:
        # 复习没通过 => 判定为遗忘,把掌握度打回,并重新进入练习流
        state.p_known = min(state.p_known, params.mastery_threshold * params.review_probability_floor)
        state.mastered_at_step = None
        _requeue_for_practice(session, concept_id)
    else:
        state.p_known = bkt_update(state.p_known, grade.score, params)

    for tag in grade.misconception_tags:
        state.misconceptions[tag] = state.misconceptions.get(tag, 0) + 1

    if is_mastered(state, concept_difficulty, params) and state.mastered_at_step is None:
        state.mastered_at_step = session.step
        schedule_review(session, concept_id, params)

    return state


def schedule_review(session: SessionState, concept_id: str, params: TutorParams = PARAMS) -> None:
    """排一次间隔复习。已排过、或复习次数已达上限的,直接跳过。

    上限很重要:没有它,「复习失败 -> 打回掌握度 -> 重新掌握 -> 又排复习」
    会形成一个自我维持的循环,课程永远结束不了。达到上限后不再复习,
    是否巩固由结课报告如实点名。
    """
    state = session.concepts.get(concept_id)
    if state and state.reviews >= params.max_reviews_per_concept:
        return
    if any(r["concept_id"] == concept_id for r in session.review_queue):
        return
    session.review_queue.append(
        {"concept_id": concept_id, "due_step": session.step + params.review_delay_steps}
    )


def advance(session: SessionState) -> None:
    """游标前进;如果当前概念没达标,顺手排一次复习。"""
    concept_id = session.current_concept_id()
    if concept_id:
        state = session.concepts.get(concept_id)
        if state and state.p_known < PARAMS.mastery_threshold:
            # 没达阈值就走了(通常是练满次数被强制推进),留一次复习兜底
            schedule_review(session, concept_id)
    session.cursor += 1
    if session.cursor >= len(session.path.ordered_concept_ids):
        session.finished = not session.review_queue


# --- 内部工具 -----------------------------------------------------------
def _due_review(session: SessionState) -> str | None:
    for entry in sorted(session.review_queue, key=lambda r: r["due_step"]):
        if entry["due_step"] <= session.step:
            session.review_queue.remove(entry)
            return entry["concept_id"]
    return None


def _dominant_misconception(state: ConceptState) -> str | None:
    if not state.misconceptions:
        return None
    tag, count = max(state.misconceptions.items(), key=lambda kv: kv[1])
    return tag if count >= 2 else None


def _requeue_for_practice(session: SessionState, concept_id: str,
                          params: TutorParams = PARAMS) -> None:
    """复习失败 -> 再给一次机会,但同样受复习次数上限约束。"""
    state = session.concepts.get(concept_id)
    if state and state.reviews >= params.max_reviews_per_concept:
        return
    if any(r["concept_id"] == concept_id for r in session.review_queue):
        return
    session.review_queue.append(
        {"concept_id": concept_id, "due_step": session.step + params.review_delay_steps}
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_report_buckets(session: SessionState, concept_difficulties: dict[str, int] | None = None,
                         params: TutorParams = PARAMS
                         ) -> tuple[list[str], list[str], list[str]]:
    difficulties = concept_difficulties or {}
    mastered, shaky, untouched = [], [], []
    for cid in session.path.ordered_concept_ids:
        st = session.concepts.get(cid)
        if st is None or st.attempts == 0:
            untouched.append(cid)
        elif is_mastered(st, difficulties.get(cid, 3), params):
            mastered.append(cid)
        else:
            shaky.append(cid)
    return mastered, shaky, untouched
