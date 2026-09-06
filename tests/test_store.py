from tutor.ingest.chunker import chunk_material
from tutor.memory.store import Store
from tutor.schemas import (ActionType, ConceptGraph, Concept, Grade, Item,
                           ItemKind, LearningPath, SessionState, Turn, Verdict)


def make_store(tmp_path) -> Store:
    return Store(tmp_path / "t.db")


def test_material_round_trip(tmp_path, sample_text):
    store = make_store(tmp_path)
    material = chunk_material("m1", "标题", sample_text)
    graph = ConceptGraph(material_id="m1", concepts=[Concept(id="k01", name="梯度")])
    path = LearningPath(material_id="m1", ordered_concept_ids=["k01"])
    store.save_material(material, graph, path)

    loaded, g, p = store.load_material("m1")
    assert len(loaded.chunks) == len(material.chunks)
    assert g.concepts[0].name == "梯度"
    assert p.ordered_concept_ids == ["k01"]


def test_saving_same_material_twice_upserts(tmp_path, sample_text):
    store = make_store(tmp_path)
    material = chunk_material("m1", "标题", sample_text)
    store.save_material(material)
    store.save_material(material)
    assert len(store.list_materials()) == 1


def test_missing_material_returns_none(tmp_path):
    assert make_store(tmp_path).load_material("nope") is None


def test_session_round_trip(tmp_path):
    store = make_store(tmp_path)
    session = SessionState(session_id="s1", material_id="m1",
                           path=LearningPath(material_id="m1", ordered_concept_ids=["k01"]),
                           step=7)
    store.save_session(session)
    session.step = 9
    store.save_session(session)
    assert store.load_session("s1").step == 9
    assert store.load_session("missing") is None


def turn(score: float, tags=()) -> Turn:
    return Turn(
        step=1, action=ActionType.PRACTICE, concept_id="k01",
        item=Item(id="i1", concept_id="k01", kind=ItemKind.MCQ, difficulty=2,
                  stem="题干", answer_key="A", source_chunk_ids=["c001"]),
        student_answer="B",
        grade=Grade(score=score,
                    verdict=Verdict.CORRECT if score >= 0.8 else Verdict.INCORRECT,
                    misconception_tags=list(tags)),
    )


def test_wrong_items_notebook(tmp_path):
    store = make_store(tmp_path)
    session = SessionState(session_id="s1", material_id="m1",
                           path=LearningPath(material_id="m1", ordered_concept_ids=["k01"]))
    store.save_session(session)
    store.record_attempt("s1", turn(0.2, ["k01-方向搞反"]))
    store.record_attempt("s1", turn(1.0))

    wrong = store.wrong_items("s1")
    assert len(wrong) == 1                        # 只收录没答好的
    assert wrong[0]["misconceptions"] == ["k01-方向搞反"]


def test_attempt_without_grade_is_skipped(tmp_path):
    store = make_store(tmp_path)
    store.record_attempt("s1", Turn(step=0, action=ActionType.TEACH, concept_id="k01"))
    assert store.wrong_items("s1") == []


def test_misconception_stats_aggregate_across_sessions(tmp_path):
    store = make_store(tmp_path)
    for sid in ("s1", "s2"):
        store.save_session(SessionState(
            session_id=sid, material_id="m1",
            path=LearningPath(material_id="m1", ordered_concept_ids=["k01"])))
        store.record_attempt(sid, turn(0.1, ["k01-方向搞反"]))
    stats = store.misconception_stats("m1")
    assert stats[0] == {"tag": "k01-方向搞反", "count": 2}
