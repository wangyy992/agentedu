"""护栏测试。安全相关的那几条是回归用例:曾经真的能读服务器任意文件。"""
import importlib

import pytest
from fastapi.testclient import TestClient

from tutor import guardrails as g


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TUTOR_FAKE_LLM", "1")
    from tutor import api, config, service
    importlib.reload(config)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    importlib.reload(service)
    importlib.reload(api)
    api.service = service.TutorService(
        llm=service.build_llm(force_fake=True), store=service.Store(tmp_path / "t.db"))
    g.reset_all(requests_per_minute=0, ingests_per_hour=0, daily_llm_calls=0)
    yield TestClient(api.app)
    g.reset_all(requests_per_minute=0, ingests_per_hour=0, daily_llm_calls=0)


# --- 安全:任意文件读取 --------------------------------------------------
def test_demo_mode_refuses_server_paths():
    """回归用例:曾经 POST {"path": "/etc/passwd"} 会把文件内容读出来返回。"""
    limits = g.Limits(demo_mode=True)
    with pytest.raises(g.MaterialRejected, match="不支持按服务器路径"):
        g.resolve_material_path("/etc/hostname", limits)


def test_material_root_blocks_traversal(tmp_path):
    allowed = tmp_path / "materials"
    allowed.mkdir()
    good = allowed / "ok.md"
    good.write_text("内容", encoding="utf-8")
    outside = tmp_path / "secret.md"
    outside.write_text("机密", encoding="utf-8")

    limits = g.Limits(demo_mode=False, material_root=str(allowed))
    assert g.resolve_material_path(str(good), limits) == good.resolve()

    for attempt in (str(outside), str(allowed / ".." / "secret.md")):
        with pytest.raises(g.MaterialRejected, match="只允许读取"):
            g.resolve_material_path(attempt, limits)


def test_missing_file_is_rejected_cleanly(tmp_path):
    limits = g.Limits(demo_mode=False)
    with pytest.raises(g.MaterialRejected, match="找不到材料文件"):
        g.resolve_material_path(str(tmp_path / "nope.md"), limits)


def test_example_name_cannot_escape_the_package():
    from tutor.examples import example_path

    for bad in ("../../etc/passwd", "a/b", ".hidden", ""):
        with pytest.raises(KeyError):
            example_path(bad)


def test_api_rejects_path_in_demo_mode(client, monkeypatch):
    monkeypatch.setattr(g, "LIMITS", g.Limits(demo_mode=True))
    r = client.post("/api/materials", json={"path": "/etc/hostname"})
    assert r.status_code == 400
    assert "服务器路径" in r.json()["detail"]


# --- 材料长度 -----------------------------------------------------------
def test_material_length_bounds():
    limits = g.Limits(min_material_chars=10, max_material_chars=50)
    assert g.validate_material_text("x" * 20, limits) == "x" * 20
    with pytest.raises(g.MaterialRejected, match="太短"):
        g.validate_material_text("x" * 5, limits)
    with pytest.raises(g.MaterialRejected, match="太长"):
        g.validate_material_text("x" * 100, limits)


def test_api_rejects_oversized_material(client, monkeypatch):
    monkeypatch.setattr(g, "LIMITS", g.Limits(max_material_chars=100, min_material_chars=10))
    r = client.post("/api/materials", json={"text": "很长的材料。" * 200})
    assert r.status_code == 400 and "太长" in r.json()["detail"]


# --- 限流 ---------------------------------------------------------------
def test_sliding_window_blocks_after_limit():
    window = g.SlidingWindow(limit=3, window_seconds=60)
    for _ in range(3):
        window.check("1.2.3.4")
    with pytest.raises(g.RateLimitExceeded) as exc:
        window.check("1.2.3.4")
    assert exc.value.retry_after > 0
    window.check("5.6.7.8")          # 不同 IP 互不影响


def test_zero_limit_means_unlimited():
    window = g.SlidingWindow(limit=0, window_seconds=60)
    for _ in range(100):
        window.check("1.2.3.4")


def test_api_returns_429_with_retry_after(client):
    g.reset_all(requests_per_minute=2)
    client.get("/api/sessions/x/next")
    client.get("/api/sessions/x/next")
    r = client.get("/api/sessions/x/next")
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) > 0


def test_ingest_has_its_own_tighter_limit(client):
    """出题限流放宽,但"导入新材料"必须单独卡死——它最贵。"""
    g.reset_all(requests_per_minute=0, ingests_per_hour=1)
    assert client.post("/api/materials", json={"example": "gradient_descent"}).status_code == 200
    assert client.post("/api/materials", json={"example": "gradient_descent"}).status_code == 429


def test_forwarded_ip_is_used_behind_a_proxy():
    class Req:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        client = type("C", (), {"host": "10.0.0.1"})()

    assert g.client_key(Req()) == "203.0.113.7"


# --- 每日预算 -----------------------------------------------------------
def test_budget_counts_and_exhausts():
    budget = g.DailyBudget(limit=3)
    assert not budget.exhausted()
    budget.record(3)
    assert budget.exhausted()
    assert budget.snapshot() == {"limit": 3, "used": 3, "exhausted": True}


def test_zero_budget_never_exhausts():
    budget = g.DailyBudget(limit=0)
    budget.record(10_000)
    assert not budget.exhausted()


def test_exhausted_budget_degrades_instead_of_failing(client):
    """预算用尽后请求仍要成功,只是换成离线模型——演示站不能看起来挂了。"""
    from tutor import api
    from tutor.llm import FakeLLM

    g.reset_all(requests_per_minute=0, ingests_per_hour=0, daily_llm_calls=1)
    g.budget.record(5)                       # 直接打满
    r = client.post("/api/materials", json={"example": "gradient_descent"})
    assert r.status_code == 200
    assert isinstance(api.service.llm, FakeLLM)
    assert client.get("/api/health").json()["budget"]["exhausted"] is True


def test_health_exposes_limits_for_the_frontend(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert "gradient_descent" in body["examples"]
    assert body["max_material_chars"] > 0
