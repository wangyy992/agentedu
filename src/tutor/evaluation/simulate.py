"""用模拟学生评测策略层。

为什么需要这个:自适应教学的效果没法靠"看几个例子"判断。这里用一个
**可控的模拟学生**代替真人,让策略层的行为变成可量化、可回归的指标。

模拟学生用 IRT(项目反应理论)的思路:
    P(答对) = c + (1-c) · sigmoid(1.7 · (θ - b))
其中 θ 是学生在该知识点上的真实能力,b 是题目难度(归一化到同一尺度),
c 是猜对概率(选择题 0.25,简答题 0.05)。讲解和补救会真的提升 θ。

关注三类指标:
1. **效率** —— 走完全程用了多少步、每个概念出了几道题;
2. **校准** —— agent 估计的掌握度 p_known 与学生真实能力 θ 的偏差;
3. **难度匹配** —— 有多少题落在「预期答对率 0.5~0.85」的甜区
   (太简单=浪费,太难=打击信心)。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..agent.orchestrator import Course, Tutor
from ..agent.policy import p_correct
from ..llm import LLMClient
from ..schemas import ActionType, ItemKind, Turn


@dataclass
class SimulatedStudent:
    """θ ∈ [0,1]:每个知识点一份真实能力,会随讲解和练习增长。"""

    ability: float = 0.55
    learn_rate: float = 0.12
    seed: int = 42
    theta: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _theta(self, concept_id: str) -> float:
        if concept_id not in self.theta:
            # 个体差异:围绕总体能力抖动
            self.theta[concept_id] = max(0.0, min(1.0, self._rng.gauss(self.ability, 0.15)))
        return self.theta[concept_id]

    def study(self, concept_id: str, intensive: bool = False) -> None:
        """听讲解 / 补救讲解带来的真实能力提升。"""
        gain = self.learn_rate * (1.6 if intensive else 1.0)
        theta = self._theta(concept_id)
        self.theta[concept_id] = min(1.0, theta + gain * (1 - theta))

    def answer(self, turn: Turn) -> tuple[str, bool]:
        """按 IRT 概率决定答对与否,并生成对应的作答文本。"""
        item = turn.item
        assert item is not None
        theta = self._theta(item.concept_id)
        b = (item.difficulty - 1) / 4.0            # 难度归一到 [0,1]
        guess = 0.25 if item.kind is ItemKind.MCQ else 0.05
        p = guess + (1 - guess) / (1 + math.exp(-1.7 * 4 * (theta - b)))
        success = self._rng.random() < p

        # 做题本身也有学习效应(答对巩固更多)
        self.theta[item.concept_id] = min(
            1.0, theta + self.learn_rate * 0.4 * (1.0 if success else 0.4) * (1 - theta)
        )

        if item.kind is ItemKind.MCQ:
            if success:
                return item.answer_key, True
            wrong = [chr(ord("A") + i) for i in range(len(item.options))
                     if chr(ord("A") + i) != item.answer_key]
            return (self._rng.choice(wrong) if wrong else "A"), False
        if success:
            return "。".join(item.rubric), True
        # 答错:只命中一半要点,模拟"部分理解"
        half = item.rubric[: max(0, len(item.rubric) // 2)]
        return ("。".join(half) if half else "我不太确定,应该和这个概念有关"), False


@dataclass
class SimResult:
    steps: int
    completed: bool
    items: int
    teaches: int
    remediations: int
    reviews: int
    items_per_concept: dict[str, int]
    calibration_mae: float
    in_sweet_spot: float
    final_overall: float
    true_ability: float
    prereq_respected: bool
    llm_calls: int
    trace: list[str] = field(default_factory=list)


def run_simulation(
    llm: LLMClient,
    course: Course,
    *,
    ability: float = 0.55,
    seed: int = 42,
    max_steps: int = 80,
) -> SimResult:
    student = SimulatedStudent(ability=ability, seed=seed)
    tutor = Tutor(llm, course)

    items = teaches = remediations = reviews = 0
    per_concept: dict[str, int] = {}
    sweet = 0
    trace: list[str] = []
    completed = False

    for _ in range(max_steps):
        turn = tutor.next_step()
        if turn.action is ActionType.COMPLETE:
            completed = True
            break

        if turn.action is ActionType.TEACH:
            teaches += 1
            student.study(turn.concept_id)
            trace.append(f"teach {turn.concept_id}")
            continue

        if turn.action is ActionType.REMEDIATE:
            remediations += 1
            student.study(turn.concept_id, intensive=True)
        elif turn.action is ActionType.REVIEW:
            reviews += 1

        # 出题前先看策略估计的答对率,判断难度是否落在甜区
        state = tutor.session.concepts[turn.concept_id]
        predicted = p_correct(state.p_known)
        if 0.5 <= predicted <= 0.85:
            sweet += 1

        answer, ok = student.answer(turn)
        done = tutor.submit_answer(answer)
        items += 1
        per_concept[turn.concept_id] = per_concept.get(turn.concept_id, 0) + 1
        trace.append(
            f"{turn.action.value} {turn.concept_id} d={turn.item.difficulty} "
            f"{'✓' if ok else '✗'} score={done.grade.score:.2f} p={state.p_known:.2f}"
        )

    # 校准误差:|agent 估计的掌握度 − 学生真实能力|
    errors = [
        abs(state.p_known - student.theta[cid])
        for cid, state in tutor.session.concepts.items()
        if cid in student.theta
    ]
    scores = [s.p_known for s in tutor.session.concepts.values()]

    return SimResult(
        steps=tutor.session.step,
        completed=completed,
        items=items,
        teaches=teaches,
        remediations=remediations,
        reviews=reviews,
        items_per_concept=per_concept,
        calibration_mae=round(sum(errors) / len(errors), 3) if errors else 0.0,
        in_sweet_spot=round(sweet / items, 3) if items else 0.0,
        final_overall=round(sum(scores) / len(scores), 3) if scores else 0.0,
        true_ability=round(
            sum(student.theta.values()) / len(student.theta), 3) if student.theta else 0.0,
        prereq_respected=_check_prereqs(course, trace),
        llm_calls=llm.usage.calls,
        trace=trace,
    )


def _check_prereqs(course: Course, trace: list[str]) -> bool:
    """硬约束核查:任何概念第一次出现时,它的先修必须都已经出现过。"""
    seen: set[str] = set()
    for line in trace:
        parts = line.split()
        if len(parts) < 2:
            continue
        cid = parts[1]
        if cid in seen:
            continue
        concept = course.concepts.get(cid)
        if concept and any(p not in seen for p in concept.prerequisites if p in course.concepts):
            return False
        seen.add(cid)
    return True


def format_result(r: SimResult) -> str:
    lines = [
        "═" * 60,
        f"模拟学生评测(真实能力均值 {r.true_ability})",
        "═" * 60,
        f"  完成           : {'是' if r.completed else '否(达到步数上限)'}",
        f"  总步数         : {r.steps}",
        f"  出题 / 讲解 / 补救 / 复习 : {r.items} / {r.teaches} / {r.remediations} / {r.reviews}",
        f"  每概念题数     : {r.items_per_concept}",
        "",
        f"  掌握度校准 MAE : {r.calibration_mae}   ← 越小说明 agent 越能看准学生",
        f"  难度落在甜区   : {r.in_sweet_spot:.0%}      ← 预期答对率 0.5~0.85 的题占比",
        f"  先修顺序合规   : {'是' if r.prereq_respected else '否 ← 规划器有 bug'}",
        f"  结课估计掌握度 : {r.final_overall}",
        "",
        f"  模型调用次数   : {r.llm_calls}",
        "═" * 60,
    ]
    return "\n".join(lines)
