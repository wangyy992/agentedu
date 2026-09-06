"""贯穿全流程的数据结构。

同时充当两个角色:
1. Python 侧的类型约束;
2. 传给 Claude 的 JSON Schema(structured outputs),保证 LLM 输出可直接解析。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- 素材与检索 ---------------------------------------------------------
class Chunk(BaseModel):
    """材料切分后的最小引用单元。出题/讲解必须引用 chunk_id 作为依据。"""

    id: str
    section: str = ""          # 所属小节标题路径,如 "3. 梯度下降 > 3.2 学习率"
    text: str
    order: int = 0             # 在原文中的顺序,用于还原上下文


class Material(BaseModel):
    id: str
    title: str
    source: str = ""
    chunks: list[Chunk] = Field(default_factory=list)

    def chunk_map(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}

    def full_text(self) -> str:
        return "\n\n".join(f"[{c.id}] {c.text}" for c in self.chunks)


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    text: str
    section: str = ""


# --- 概念图与学习路径 ---------------------------------------------------
class Concept(BaseModel):
    """从材料里抽出的知识点。prerequisites 构成有向图,用于规划学习顺序。"""

    id: str
    name: str
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)  # 其它 concept 的 id
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    difficulty: int = 2  # 1~5,概念本身的固有难度


class ConceptGraph(BaseModel):
    material_id: str
    concepts: list[Concept] = Field(default_factory=list)

    def by_id(self) -> dict[str, Concept]:
        return {c.id: c for c in self.concepts}


class LearningPath(BaseModel):
    """规划器的输出:一个可执行的教学顺序 + 给人看的理由。"""

    material_id: str
    ordered_concept_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    cycles_broken: list[str] = Field(default_factory=list)  # 记录被打破的循环依赖


# --- 题目与判分 ---------------------------------------------------------
class ItemKind(str, Enum):
    MCQ = "mcq"          # 单选
    SHORT = "short"      # 简答
    CLOZE = "cloze"      # 填空


class Item(BaseModel):
    """一道题。source_chunk_ids 是 RAG 依据,判分和申诉时都要用到。"""

    id: str = ""
    concept_id: str = ""
    kind: ItemKind = ItemKind.SHORT
    difficulty: int = 2
    stem: str                                   # 题干
    options: list[str] = Field(default_factory=list)  # 仅 mcq 使用,形如 "A. ..."
    answer_key: str                             # mcq 填选项字母;简答填参考答案
    rubric: list[str] = Field(default_factory=list)   # 评分要点,判分时逐条核对
    rationale: str = ""                          # 为什么这是对的(答对时直接复用)
    hint: str = ""                               # 答错时的第一级提示
    source_chunk_ids: list[str] = Field(default_factory=list)


class Verdict(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"


class Grade(BaseModel):
    """判分结果。misconception_tags 是自适应的关键输入——决定补救讲什么。"""

    score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    feedback: str = ""
    matched_rubric: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    misconception_tags: list[str] = Field(default_factory=list)


# --- 会话状态 -----------------------------------------------------------
class ConceptState(BaseModel):
    """单个概念的掌握状态。p_known 由 BKT 更新,是自适应的核心状态。"""

    concept_id: str
    p_known: float = 0.15
    attempts: int = 0
    correct: int = 0
    taught: bool = False
    difficulty: int = 1                        # 当前投放难度
    misconceptions: dict[str, int] = Field(default_factory=dict)
    reviews: int = 0                            # 已做过的间隔复习次数(上限见 TutorParams)
    max_difficulty_seen: int = 0                # 做过的最高难度,达标需要够高的难度背书
    last_step: int = -1
    mastered_at_step: int | None = None


class ActionType(str, Enum):
    TEACH = "teach"            # 讲解
    PRACTICE = "practice"      # 常规练习
    REMEDIATE = "remediate"    # 针对误区的补救讲解 + 降难度题
    REVIEW = "review"          # 间隔复习旧概念
    ADVANCE = "advance"        # 推进到下一个概念
    COMPLETE = "complete"      # 全部完成


class StepPlan(BaseModel):
    """策略层的决策:下一步做什么、对哪个概念、什么难度、为什么。"""

    action: ActionType
    concept_id: str | None = None
    difficulty: int = 2
    reason: str = ""
    focus_misconception: str | None = None


class Turn(BaseModel):
    """一次交互的完整留痕,便于回放与评测。"""

    step: int
    action: ActionType
    concept_id: str | None = None
    content: str = ""                 # 讲解正文(teach/remediate)
    item: Item | None = None
    student_answer: str | None = None
    grade: Grade | None = None
    citations: list[str] = Field(default_factory=list)
    reason: str = ""


class SessionState(BaseModel):
    session_id: str
    material_id: str
    path: LearningPath
    cursor: int = 0
    step: int = 0
    concepts: dict[str, ConceptState] = Field(default_factory=dict)
    review_queue: list[dict[str, Any]] = Field(default_factory=list)  # {concept_id, due_step}
    pending: Turn | None = None       # 已发出但尚未作答的题
    history: list[Turn] = Field(default_factory=list)
    finished: bool = False

    def current_concept_id(self) -> str | None:
        ids = self.path.ordered_concept_ids
        return ids[self.cursor] if 0 <= self.cursor < len(ids) else None


# --- 答疑(tool-use loop 的输出) --------------------------------------
class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    used_searches: list[str] = Field(default_factory=list)
    grounded: bool = True   # 材料里找不到依据时为 False,并在 answer 中说明


# --- 报告 ---------------------------------------------------------------
class MasteryReport(BaseModel):
    session_id: str
    overall: float = 0.0
    mastered: list[str] = Field(default_factory=list)
    shaky: list[str] = Field(default_factory=list)
    not_started: list[str] = Field(default_factory=list)
    top_misconceptions: list[dict[str, Any]] = Field(default_factory=list)
    narrative: str = ""


# --- JSON Schema 工具 ---------------------------------------------------
def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """把 pydantic 模型转成 Claude structured outputs 能接受的严格 schema。

    要求:每层对象都要 additionalProperties: false,且所有字段进 required
    (有默认值的字段 pydantic 不会放进 required,但严格模式要求全列出)。
    """
    schema = model.model_json_schema()
    _tighten(schema)
    return schema


def _tighten(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        # 严格模式只认 schema 关键字的一个子集。数值区间等约束在客户端用
        # pydantic 兜底校验,这里剔掉,避免请求被 400。
        for key in ("default", "$comment", "minimum", "maximum",
                    "exclusiveMinimum", "exclusiveMaximum", "format"):
            node.pop(key, None)
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for value in node:
            _tighten(value)


__all__ = [n for n in dir() if not n.startswith("_")]
