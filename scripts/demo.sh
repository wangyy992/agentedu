#!/usr/bin/env bash
# 一键演示:离线模式跑完整链路,不需要 API Key。
set -euo pipefail
cd "$(dirname "$0")/.."

export TUTOR_FAKE_LLM=1
PY="${PYTHON:-python3}"

echo "════════ ① 理解材料 → 知识图谱 → 学习路径 ════════"
$PY -m tutor.cli --fake ingest examples/gradient_descent.md

echo
echo "════════ ② 模拟三种水平的学生,看自适应差异 ════════"
for ability in 0.25 0.55 0.90; do
  echo "--- 学生能力 $ability ---"
  $PY -m tutor.cli --fake eval examples/gradient_descent.md --ability "$ability"
done

echo
echo "接下来可以试试:"
echo "  $PY -m tutor.cli --fake tutor examples/gradient_descent.md   # 交互式辅导"
echo "  $PY -m tutor.cli --fake serve                                # 网页界面"
