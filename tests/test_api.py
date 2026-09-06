import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TUTOR_FAKE_LLM", "1")
    import importlib

    from tutor import api, config, guardrails, service
    importlib.reload(config)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    importlib.reload(service)
    importlib.reload(api)
    # 显式注入离线模型,避免依赖 import 顺序和环境变量的读取时机
    api.service = service.TutorService(
        llm=service.build_llm(force_fake=True), store=service.Store(tmp_path / "t.db")
    )
    # 护栏是模块级单例,会跨测试累计。这里清零并放开上限,
    # 让功能测试不受限流干扰;限流本身由 test_guardrails.py 专门覆盖。
    guardrails.reset_all(requests_per_minute=0, ingests_per_hour=0, daily_llm_calls=0)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_full_flow(client):
    r = client.post("/api/materials", json={"example": "gradient_descent"})
    assert r.status_code == 200
    material = r.json()
    assert material["concepts"] and material["path"]

    sid = client.post("/api/sessions", json={"material_id": material["material_id"]}).json()["session_id"]

    answered = 0
    for _ in range(40):
        turn = client.get(f"/api/sessions/{sid}/next").json()
        assert "reason" in turn                 # 每一步都要能解释自己
        if turn["action"] == "complete":
            break
        if turn["item"]:
            g = client.post(f"/api/sessions/{sid}/answer", json={"answer": "A"}).json()
            assert 0.0 <= g["score"] <= 1.0
            assert 0.0 <= g["mastery"] <= 1.0
            answered += 1
    assert answered > 0

    report = client.get(f"/api/sessions/{sid}/report").json()
    assert 0.0 <= report["overall"] <= 1.0
    assert "wrong_items" in report

    state = client.get(f"/api/sessions/{sid}/state").json()
    assert state["history"] and state["usage"]["calls"] > 0


def test_answer_without_pending_item_is_409(client):
    material = client.post("/api/materials", json={"example": "gradient_descent"}).json()
    sid = client.post("/api/sessions", json={"material_id": material["material_id"]}).json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/answer", json={"answer": "A"}).status_code == 409


def test_unknown_ids_are_404(client):
    assert client.post("/api/sessions", json={"material_id": "nope"}).status_code == 404
    assert client.get("/api/sessions/nope/next").status_code == 404


def test_unknown_example_is_400(client):
    assert client.post("/api/materials", json={"example": "nope"}).status_code == 400


def test_ask_returns_verified_citations(client):
    material = client.post("/api/materials", json={"example": "gradient_descent"}).json()
    sid = client.post("/api/sessions", json={"material_id": material["material_id"]}).json()["session_id"]
    a = client.post(f"/api/sessions/{sid}/ask", json={"question": "学习率过大会怎样"}).json()
    assert a["citations"] and a["grounded"] is True
    assert a["used_searches"]
