#!/usr/bin/env bash
# 一键环境配置。可以反复执行,不会重复干活。
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m错误:%s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. 找一个够新的 Python -------------------------------------------
PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done
[ -n "$PY" ] || die "需要 Python 3.10 或更高版本。
  macOS:  brew install python@3.12
  Ubuntu: sudo apt install python3.12 python3.12-venv
  Windows: 到 python.org 下载安装,并勾选 Add Python to PATH"

say "① 使用 $("$PY" --version 2>&1)"

# --- 2. 虚拟环境 -------------------------------------------------------
if [ ! -d .venv ]; then
  say "② 创建虚拟环境 .venv"
  "$PY" -m venv .venv || die "创建虚拟环境失败。Ubuntu 上可能需要先装 python3-venv"
else
  say "② 虚拟环境已存在,跳过"
fi

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"   # Git Bash on Windows
[ -x "$VENV_PY" ] || die "找不到虚拟环境里的 python"

# --- 3. 装依赖 ---------------------------------------------------------
say "③ 安装依赖(第一次会慢一点)"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -e ".[pdf,dev]" || die "依赖安装失败,请把上面的报错发出来"

# --- 4. API Key --------------------------------------------------------
say "④ 配置 API Key"
if [ -f .env ] && grep -qE '^\s*ANTHROPIC_API_KEY=sk-' .env; then
  echo "   .env 里已经有 key 了,跳过。要换 key 就直接编辑 .env"
else
  echo "   没有 key 也能用(离线模式),但讲解和判分质量会很差。"
  echo "   Key 从 https://console.anthropic.com/settings/keys 获取。"
  printf "   粘贴你的 ANTHROPIC_API_KEY(直接回车 = 先跳过):"
  read -r key || key=""
  if [ -n "$key" ]; then
    [ -f .env ] || cp .env.example .env
    # 覆盖已有行;没有则追加
    if grep -q '^ANTHROPIC_API_KEY=' .env 2>/dev/null; then
      "$VENV_PY" - "$key" <<'PYEOF'
import re, sys, pathlib
key = sys.argv[1]
p = pathlib.Path(".env")
text = re.sub(r"(?m)^ANTHROPIC_API_KEY=.*$", f"ANTHROPIC_API_KEY={key}", p.read_text())
p.write_text(text)
PYEOF
    else
      printf '\nANTHROPIC_API_KEY=%s\n' "$key" >> .env
    fi
    echo "   已写入 .env(该文件在 .gitignore 里,不会被提交)"
  else
    warn "   已跳过。之后可以编辑 .env 补上,或者加 --fake 用离线模式。"
  fi
fi

# --- 5. 自检 -----------------------------------------------------------
say "⑤ 自检"
"$VENV_PY" -m pytest -q 2>&1 | grep -E "passed|failed|error" | tail -1 \
  || warn "   自检没跑起来,但不影响使用"

HAS_KEY=$("$VENV_PY" -c "
import sys; sys.path.insert(0,'src')
from tutor import config   # 会加载 .env
import os
print('yes' if os.getenv('ANTHROPIC_API_KEY','').startswith('sk-') else 'no')
" 2>/dev/null || echo no)

say "完成。接下来:"
echo
echo "   source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate"
if [ "$HAS_KEY" = "yes" ]; then
  echo "   python -m tutor.cli serve      # 网页界面 → http://127.0.0.1:8000"
  echo "   python -m tutor.cli tutor gradient_descent    # 命令行辅导"
  echo
  echo "   想用自己的材料:python -m tutor.cli tutor 你的讲义.pdf"
else
  warn "   当前没有 API Key,只能用离线模式(内容质量很差,仅用于验证流程):"
  echo "   python -m tutor.cli --fake serve"
fi
