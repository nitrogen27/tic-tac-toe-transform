#!/bin/bash
# Generate TypeScript and Python types from JSON Schema definitions
# Usage: bash packages/shared/codegen.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_DIR="$SCRIPT_DIR/schemas"
TS_OUT="$SCRIPT_DIR/generated/typescript"
PY_OUT="$SCRIPT_DIR/generated/python"

echo "[codegen] Generating TypeScript types..."
mkdir -p "$TS_OUT"
if command -v npx &>/dev/null; then
  for schema in "$SCHEMA_DIR"/*.schema.json; do
    name=$(basename "$schema" .schema.json)
    npx json-schema-to-typescript "$schema" -o "$TS_OUT/${name}.d.ts" --no-banner 2>/dev/null || \
      echo "  WARN: skipped $name (json-schema-to-typescript not installed)"
  done
  echo "[codegen] TypeScript types written to $TS_OUT/"
else
  echo "[codegen] SKIP TypeScript (npx not found)"
fi

echo "[codegen] Generating Python Pydantic models..."
mkdir -p "$PY_OUT"
if command -v datamodel-codegen &>/dev/null; then
  for schema in "$SCHEMA_DIR"/*.schema.json; do
    name=$(basename "$schema" .schema.json)
    datamodel-codegen --input "$schema" --output "$PY_OUT/${name}.py" \
      --output-model-type pydantic_v2.BaseModel \
      --target-python-version 3.11 2>/dev/null || \
      echo "  WARN: skipped $name (datamodel-codegen failed)"
  done
  echo "[codegen] Python models written to $PY_OUT/"
else
  echo "[codegen] SKIP Python (datamodel-codegen not installed)"
  echo "[codegen] Install with: pip install datamodel-code-generator"
fi

echo "[codegen] Done."
