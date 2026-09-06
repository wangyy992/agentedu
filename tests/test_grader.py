from tutor.agent.grader import grade
from tutor.ingest.chunker import chunk_material
from tutor.llm import FakeLLM
from tutor.retrieval.bm25 import BM25Retriever
from tutor.schemas import Item, ItemKind, Verdict


def mcq(**kw) -> Item:
    base = dict(id="i1", concept_id="k01", kind=ItemKind.MCQ, difficulty=2,
                stem="下列哪个正确?", options=["A. 对的", "B. 错的", "C. 无关", "D. 也错"],
                answer_key="A", rubric=["选出正确项"], rationale="因为材料这么说",
                source_chunk_ids=["c001"])
    base.update(kw)
    return Item(**base)


def retriever(sample_text):
    return BM25Retriever(chunk_material("m", "t", sample_text).chunks)


def test_correct_mcq_short_circuits_without_calling_the_model(sample_text):
    llm = FakeLLM()
    result = grade(llm, retriever(sample_text), mcq(), "A", "梯度下降")
    assert result.verdict is Verdict.CORRECT and result.score == 1.0
    assert llm.usage.calls == 0          # 规则判定,零 token
    assert "因为材料这么说" in result.feedback


def test_choice_letter_is_parsed_from_free_text(sample_text):
    for answer in ("A", "a", "我选 A", "A. 对的", "答案是a)"):
        result = grade(FakeLLM(), retriever(sample_text), mcq(), answer, "梯度下降")
        assert result.verdict is Verdict.CORRECT, answer


def test_wrong_mcq_gets_no_partial_credit(sample_text):
    """选择题选错就是错——不能因为"思路不错"给部分分。"""
    result = grade(FakeLLM(), retriever(sample_text), mcq(), "B", "梯度下降")
    assert result.score <= 0.2 and result.verdict is Verdict.INCORRECT


def test_empty_answer_is_zero_and_costs_nothing(sample_text):
    llm = FakeLLM()
    result = grade(llm, retriever(sample_text), mcq(), "   ", "梯度下降")
    assert result.score == 0.0 and result.verdict is Verdict.INCORRECT
    assert llm.usage.calls == 0
    assert result.missing_points == ["选出正确项"]


def test_short_answer_is_graded_by_rubric(sample_text):
    item = Item(id="i2", concept_id="k01", kind=ItemKind.SHORT, difficulty=3,
                stem="解释梯度下降", answer_key="沿负梯度更新",
                rubric=["提到 负梯度", "提到 迭代"], source_chunk_ids=["c002"])
    good = grade(FakeLLM(), retriever(sample_text), item, "沿着负梯度方向迭代更新参数", "梯度下降")
    bad = grade(FakeLLM(), retriever(sample_text), item, "不知道", "梯度下降")
    assert good.score > bad.score
    assert bad.misconception_tags        # 答错必须留下误区标签,否则自适应没有燃料


def test_correct_answers_never_carry_misconception_tags(sample_text):
    item = Item(id="i3", concept_id="k01", kind=ItemKind.SHORT, difficulty=2,
                stem="?", answer_key="x", rubric=["提到 梯度"], source_chunk_ids=["c001"])
    result = grade(FakeLLM(), retriever(sample_text), item, "梯度", "梯度下降")
    assert result.verdict is Verdict.CORRECT and result.misconception_tags == []
