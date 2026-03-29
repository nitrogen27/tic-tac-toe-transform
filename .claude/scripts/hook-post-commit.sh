#!/usr/bin/env bash
# hook-post-commit.sh — выводит краткий статус после коммита
# Подключить как git hook: cp .claude/scripts/hook-post-commit.sh .git/hooks/post-commit
# Или запускать вручную: bash .claude/scripts/hook-post-commit.sh

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git log -1 --pretty="%h %s")
FILES=$(git diff-tree --no-commit-id -r --name-only HEAD | wc -l | tr -d ' ')

echo ""
echo "┌─ Post-commit ───────────────────────────────────────┐"
echo "│ Branch:  $BRANCH"
echo "│ Commit:  $COMMIT"
echo "│ Files:   $FILES changed"
echo "└──────────────────────────────────────────────────────┘"
echo ""

# Напомнить запушить если нет remote tracking
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
if [ -z "$REMOTE" ] || [ "$LOCAL" != "$REMOTE" ]; then
  echo "  → Не запушено. Запустить: git push origin $BRANCH"
fi
echo ""
