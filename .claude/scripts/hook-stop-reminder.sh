#!/usr/bin/env bash
# hook-stop-reminder.sh — напоминание при остановке работы
# Показывает незавершённые задачи из MVP_ROADMAP.md
# Использование: bash .claude/scripts/hook-stop-reminder.sh

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo ""
echo "┌─ Статус MVP Gomoku Platform V3 ─────────────────────┐"

# Фазы завершены?
check_phase() {
  local phase=$1
  local label=$2
  if grep -q "✅ Phase $phase" docs/gomoku-platform-v3/README.md 2>/dev/null; then
    echo "│  ✅ Phase $phase — $label"
  elif grep -q "🔲 Phase $phase" docs/gomoku-platform-v3/README.md 2>/dev/null; then
    echo "│  🔲 Phase $phase — $label"
  else
    echo "│  ?  Phase $phase — $label"
  fi
}

check_phase 1 "Monorepo Scaffolding"
check_phase 2 "C++17 Engine Core"
check_phase 3 "PyTorch Trainer"
check_phase 4 "FastAPI Gateway"
check_phase 5 "Web MVP"
check_phase 6 "Self-Play Pipeline"

echo "└──────────────────────────────────────────────────────┘"
echo ""

# Незапушенные коммиты
UNPUSHED=$(git log @{u}..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNPUSHED" -gt "0" ]; then
  echo "  ⚠ $UNPUSHED незапушенных коммита:"
  git log @{u}..HEAD --oneline 2>/dev/null | sed 's/^/    /'
  echo ""
fi

# Незафиксированные изменения
DIRTY=$(git status --short | grep -v "^??" | grep -v __pycache__ | grep -v ".pyc" | wc -l | tr -d ' ')
if [ "$DIRTY" -gt "0" ]; then
  echo "  ⚠ $DIRTY незафиксированных изменений (git status)"
fi

# Следующая задача из roadmap
NEXT=$(grep -m1 "^- \[ \]" docs/gomoku-platform-v3/MVP_ROADMAP.md 2>/dev/null | sed 's/^- \[ \] //')
if [ -n "$NEXT" ]; then
  echo "  → Следующий чекпойнт: $NEXT"
fi
echo ""
