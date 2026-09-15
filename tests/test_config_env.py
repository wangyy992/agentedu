"""`.env` 加载。

单独测是因为这里最容易出「配了但不生效」的静默故障:
.env.example 摆在仓库里,用户照着建了 .env,程序却读不到——
排查起来非常费时,而且第一反应通常是怀疑 key 本身错了。
"""
import importlib
import os

from tutor import config


def reload_config(monkeypatch, cwd, **env):
    monkeypatch.chdir(cwd)
    for k in ("ANTHROPIC_API_KEY", "TUTOR_MODEL", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(config)


def test_reads_key_from_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-xyz\n", encoding="utf-8")
    reload_config(monkeypatch, tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-xyz"


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    """Docker / CI 里传进来的值不能被仓库里的 .env 覆盖。"""
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    reload_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="from-shell")
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_handles_comments_quotes_and_export(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# 这是注释\n\n"
        'export TUTOR_MODEL="claude-opus-5"\n'
        "ANTHROPIC_API_KEY='sk-ant-quoted'\n"
        "格式不对的一行\n",
        encoding="utf-8",
    )
    cfg = reload_config(monkeypatch, tmp_path)
    assert cfg.MODEL == "claude-opus-5"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-quoted"


def test_no_dotenv_is_fine(tmp_path, monkeypatch):
    cfg = reload_config(monkeypatch, tmp_path)
    assert cfg.MODEL                      # 走默认值,不报错
