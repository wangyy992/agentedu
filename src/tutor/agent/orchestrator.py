"""编排层:把"理解 -> 规划 -> 讲解 -> 出题 -> 判分 -> 自适应"串成一个有状态的循环。

对外只暴露三个动词:
    prepare_course(material)          一次性:理解材料 + 规划路径
    Tutor.next_step()                 拿到下一步(讲解 或 一道题)
    Tutor.submit_answer(answer)       提交作答 -> 判分 -> 更新状态

设计上刻意让「决策」和「生成」分离:
    policy.decide()  决定做什么(纯函数、可测)
    本模块           负责把决策变成一次具体的模型调用
这样任何一次教学行为都能回答"为什么是这一步"(StepPlan.reason)。
"""
from __future__ import annotations

import logging
import uuid

from ..config import PARAMS
from ..llm import LLMClient, Prompt
from ..retrieval.bm25 import BM25Retriever
from ..schemas import (
    ActionType,
    Concept,
    ConceptGraph,
    ConceptState,
    Grade,
    GroundedAnswer,
    LearningPath,
    MasteryReport,
    Material,
    SessionState,
    Turn,
)
from . import policy
from .comprehend import extract_concepts
from .explainer import explain, remediate
from .grader import grade as grade_answer
from .item_writer import write_item
from .planner import plan_path
from .qa import answer_question

log = logging.getLogger("tutor.orchestrator")

REPORT_SYSTEM = """你在写一份学习小结,给学生本人看。要求:
1. 先一句话总评(具体,不要"表现不错"这种空话)。
2. 分三块:已掌握 / 还需巩固 / 下一步建议。
3. 「还需巩固」要点名具体的误区,并给出一条可操作的行动(比如重读哪一节、重做哪类题)。
4. 200~300 字,不要客套。"""


class Course:
    """一份材料的静态产物:切好的块、知识图谱、学习路径、检索器。可复用于多个学生。"""

    def __init__(self, material: Material, graph: ConceptGraph, path: LearningPath) -> None:
        self.material = material
        self.graph = graph
        self.path = path
        self.retriever = BM25Retriever(material.chunks)
        self._concepts = graph.by_id()

    @property
    def concepts(self) -> dict[str, Concept]:
        return self._concepts

    def concept(self, cid: str) -> Concept:
        return self._concepts[cid]

    def difficulties(self) -> dict[str, int]:
        return {cid: c.difficulty for cid, c in self._concepts.items()}

    def names(self) -> dict[str, str]:
        return {cid: c.name for cid, c in self._concepts.items()}

    def cached_context(self) -> str:
        """放进 prompt cache 的那段材料全文。"""
        return self.material.full_text()


def prepare_course(llm: LLMClient, material: Material) -> Course:
    """一次性预处理:读懂材料 -> 建知识图谱 -> 排出学习路径。"""
    graph = extract_concepts(llm, material)
    if not graph.concepts:
        raise ValueError("未能从材料中抽取到任何知识点,请检查材料内容是否过短或为空")
    path = plan_path(graph)
    log.info("材料 %s:%d 个知识点,路径 %s", material.id, len(graph.concepts), path.ordered_concept_ids)
    return Course(material, graph, path)


class Tutor:
    """一个学生 + 一份课程 = 一个 Tutor 实例。所有状态都在 self.session 里。"""

    def __init__(self, llm: LLMClient, course: Course, session: SessionState | None = None) -> None:
        self.llm = llm
        self.course = course
        self.session = session or SessionState(
            session_id=uuid.uuid4().hex[:12],
            material_id=course.material.id,
            path=course.path,
            concepts={
                cid: ConceptState(concept_id=cid, p_known=PARAMS.p_init)
                for cid in course.path.ordered_concept_ids
            },
        )

    # -- 主循环 ---------------------------------------------------------
    def next_step(self) -> Turn:
        """推进到下一个动作。如果上一题还没作答,原样返回那道题。"""
        if self.session.pending is not None:
            return self.session.pending
        if self.session.finished:
            return self._completed_turn()

        # ADVANCE 只是游标移动,不消耗一个"步",所以在循环里连续消化掉
        for _ in range(len(self.session.path.ordered_concept_ids) + 2):
            plan = policy.decide(self.session, self.course.difficulties())
            if plan.action is ActionType.ADVANCE:
                log.info("推进:%s(%s)", plan.concept_id, plan.reason)
                policy.advance(self.session)
                if self.session.current_concept_id() is None and not self.session.review_queue:
                    self.session.finished = True
                    return self._completed_turn()
                continue
            if plan.action is ActionType.COMPLETE:
                if self.session.review_queue:
                    # 正课走完但还有复习债,把游标停住,交给复习队列收尾
                    self.session.review_queue.sort(key=lambda r: r["due_step"])
                    self.session.review_queue[0]["due_step"] = self.session.step
                    continue
                self.session.finished = True
                return self._completed_turn()
            return self._execute(plan)

        self.session.finished = True
        return self._completed_turn()

    def submit_answer(self, answer: str) -> Turn:
        """判分 + 回写状态。返回的是**已完成**的这一轮(含 grade)。"""
        turn = self.session.pending
        if turn is None or turn.item is None:
            raise ValueError("当前没有待作答的题目,请先调用 next_step()")

        concept = self.course.concept(turn.item.concept_id)
        result: Grade = grade_answer(
            self.llm,
            self.course.retriever,
            turn.item,
            answer,
            concept.name,
            cached_context=self.course.cached_context(),
        )

        turn.student_answer = answer
        turn.grade = result
        is_review = turn.action is ActionType.REVIEW
        state = policy.apply_grade(
            self.session, concept.id, result, is_review=is_review,
            item_difficulty=turn.item.difficulty, concept_difficulty=concept.difficulty,
        )
        state.difficulty = turn.item.difficulty

        self.session.pending = None
        self.session.history.append(turn)
        self.session.step += 1
        log.info(
            "判分 %s: score=%.2f 掌握度 %.2f 误区=%s",
            concept.name, result.score, state.p_known, result.misconception_tags,
        )
        return turn

    def ask(self, question: str) -> GroundedAnswer:
        """随时打断提问。不影响教学状态,但模型可以查到学生的掌握情况。"""
        return answer_question(
            self.llm,
            self.course.retriever,
            self.course.material,
            question,
            session=self.session,
            concept_names=self.course.names(),
            cached_context=self.course.cached_context(),
        )

    # -- 把决策变成一次具体调用 ------------------------------------------
    def _execute(self, plan) -> Turn:
        concept = self.course.concept(plan.concept_id)
        ctx = self.course.cached_context()

        if plan.action is ActionType.TEACH:
            body, cites = explain(self.llm, self.course.retriever, concept, cached_context=ctx)
            self.session.concepts[concept.id].taught = True
            turn = Turn(
                step=self.session.step, action=plan.action, concept_id=concept.id,
                content=body, citations=cites, reason=plan.reason,
            )
            self.session.history.append(turn)
            self.session.step += 1
            return turn

        if plan.action is ActionType.REMEDIATE:
            state = self.session.concepts[concept.id]
            body, cites = remediate(
                self.llm, self.course.retriever, concept, state,
                plan.focus_misconception or "", cached_context=ctx,
            )
            # 补救 = 重讲 + 立刻用一道降难度的题验证,所以这一轮同时带 content 和 item
            item = write_item(
                self.llm, self.course.retriever, concept, plan.difficulty,
                asked_stems=self._asked_stems(concept.id),
                focus_misconception=plan.focus_misconception,
                cached_context=ctx,
            )
            item.id = self._item_id()
            turn = Turn(
                step=self.session.step, action=plan.action, concept_id=concept.id,
                content=body, item=item, citations=cites, reason=plan.reason,
            )
            self.session.pending = turn
            return turn

        # PRACTICE / REVIEW
        item = write_item(
            self.llm, self.course.retriever, concept, plan.difficulty,
            asked_stems=self._asked_stems(concept.id), cached_context=ctx,
        )
        item.id = self._item_id()
        turn = Turn(
            step=self.session.step, action=plan.action, concept_id=concept.id,
            item=item, citations=item.source_chunk_ids, reason=plan.reason,
        )
        self.session.pending = turn
        return turn

    # -- 结课 -----------------------------------------------------------
    def _completed_turn(self) -> Turn:
        report = self.report()
        return Turn(
            step=self.session.step, action=ActionType.COMPLETE,
            content=report.narrative, reason="全部知识点已完成",
        )

    def report(self) -> MasteryReport:
        mastered, shaky, untouched = policy.build_report_buckets(
            self.session, self.course.difficulties())
        names = self.course.names()

        tally: dict[str, int] = {}
        for state in self.session.concepts.values():
            for tag, count in state.misconceptions.items():
                tally[tag] = tally.get(tag, 0) + count
        top = [
            {"tag": tag, "count": count}
            for tag, count in sorted(tally.items(), key=lambda kv: -kv[1])[:5]
        ]

        scores = [s.p_known for s in self.session.concepts.values()]
        overall = sum(scores) / len(scores) if scores else 0.0

        misconception_summary = "; ".join(f"{t['tag']}×{t['count']}" for t in top)

        prompt = Prompt(
            task="final_report",
            system=REPORT_SYSTEM,
            user=(
                f"材料:{self.course.material.title}\n"
                f"已掌握:{'、'.join(names.get(c, c) for c in mastered) or '无'}\n"
                f"待巩固:{'、'.join(names.get(c, c) for c in shaky) or '无'}\n"
                f"未开始:{'、'.join(names.get(c, c) for c in untouched) or '无'}\n"
                f"高频误区:{misconception_summary or '无'}\n"
                f"共作答 {sum(s.attempts for s in self.session.concepts.values())} 题。\n"
                "请写学习小结。"
            ),
            data={
                "total": len(self.session.concepts),
                "mastered": mastered,
                "shaky": shaky,
            },
        )
        narrative = self.llm.text(prompt, cached_context=self.course.cached_context())

        return MasteryReport(
            session_id=self.session.session_id,
            overall=round(overall, 3),
            mastered=[names.get(c, c) for c in mastered],
            shaky=[names.get(c, c) for c in shaky],
            not_started=[names.get(c, c) for c in untouched],
            top_misconceptions=top,
            narrative=narrative,
        )

    # -- 小工具 ---------------------------------------------------------
    def _asked_stems(self, concept_id: str) -> list[str]:
        return [
            t.item.stem for t in self.session.history
            if t.item is not None and t.item.concept_id == concept_id
        ]

    def _item_id(self) -> str:
        return f"i{len(self.session.history) + 1:03d}"
