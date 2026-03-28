SHELL := /bin/bash

.PHONY: help codegen build-engine test-engine bench-engine api web train dev-up dev-down legacy-up legacy-down clean

help:
	@echo "Gomoku Platform V3 — Build Targets"
	@echo ""
	@echo "  Schemas:"
	@echo "    make codegen         - Generate TS/Python types from JSON schemas"
	@echo ""
	@echo "  C++ Engine:"
	@echo "    make build-engine    - Build C++ engine (CMake + vcpkg)"
	@echo "    make test-engine     - Run engine unit tests"
	@echo "    make bench-engine    - Run engine benchmarks"
	@echo ""
	@echo "  Services:"
	@echo "    make api             - Start FastAPI dev server (port 8000)"
	@echo "    make web             - Start React dev server (port 5174)"
	@echo "    make train           - Run PyTorch training (requires GPU)"
	@echo ""
	@echo "  Docker:"
	@echo "    make dev-up          - Start new platform services"
	@echo "    make dev-down        - Stop new platform services"
	@echo "    make legacy-up       - Start legacy server+client (GPU)"
	@echo "    make legacy-down     - Stop legacy services"

codegen:
	bash packages/shared/codegen.sh

build-engine:
	cmake -B engine-core/build -S engine-core \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build engine-core/build --config Release -j4

test-engine: build-engine
	cd engine-core/build && ctest --output-on-failure

bench-engine: build-engine
	./engine-core/build/gomoku_bench

api:
	cd apps/api && PYTHONPATH=src uvicorn gomoku_api.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd apps/web && npm run dev

train:
	cd trainer-lab && PYTHONPATH=src python -m trainer_lab.training.trainer

dev-up:
	docker compose -f docker-compose.dev.yml up -d --build

dev-down:
	docker compose -f docker-compose.dev.yml down

legacy-up:
	docker compose -f docker-compose.gpu.yml up -d --build

legacy-down:
	docker compose -f docker-compose.gpu.yml down

clean:
	rm -rf engine-core/build
	rm -rf apps/web/node_modules apps/web/dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
