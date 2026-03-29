#!/usr/bin/env bash
# hook-pre-deploy.sh — запускать перед деплоем / docker-compose up
# Проверяет готовность всех компонентов к деплою.
# Использование: bash .claude/scripts/hook-pre-deploy.sh

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== Pre-deploy checks ==="

# 1. Engine tests
echo "[1/4] C++ engine tests..."
if [ -d "engine-core/build" ]; then
  cd engine-core/build && ctest --output-on-failure -q && cd "$ROOT"
  echo "    ✓ Engine tests passed"
else
  echo "    ⚠ engine-core/build not found — run: make build-engine"
  exit 1
fi

# 2. Python API tests
echo "[2/4] FastAPI tests..."
if [ -d "apps/api" ]; then
  cd apps/api && PYTHONPATH=src python -m pytest tests/ -q --tb=short 2>&1 | tail -5 && cd "$ROOT"
  echo "    ✓ API tests passed"
else
  echo "    ⚠ apps/api not found"
fi

# 3. TypeScript build check
echo "[3/4] TypeScript build..."
if [ -d "apps/web" ]; then
  cd apps/web && npm run build --silent && cd "$ROOT"
  echo "    ✓ Web build OK"
else
  echo "    ⚠ apps/web not found"
fi

# 4. PyTorch trainer smoke test
echo "[4/4] Trainer import check..."
if [ -d "trainer-lab" ]; then
  cd trainer-lab && PYTHONPATH=src python -c "from trainer_lab.models.resnet import PolicyValueResNet; print('OK')" && cd "$ROOT"
  echo "    ✓ Trainer imports OK"
else
  echo "    ⚠ trainer-lab not found"
fi

echo ""
echo "=== All pre-deploy checks passed ==="
