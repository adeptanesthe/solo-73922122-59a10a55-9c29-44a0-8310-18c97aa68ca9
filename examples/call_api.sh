#!/usr/bin/env bash
# 调用漏检复核 API 的示例。用法：
#   ./examples/call_api.sh [BASE_URL] [REQUEST_JSON]
# 默认 BASE_URL=http://localhost:8000，REQUEST_JSON=examples/request_solved.json
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
REQUEST_JSON="${2:-examples/request_solved.json}"

echo "== GET  /health =="
curl -sS "$BASE_URL/health"
echo
echo "== POST /api/v1/review  ($REQUEST_JSON) =="
curl -sS -X POST "$BASE_URL/api/v1/review" \
  -H 'Content-Type: application/json' \
  -d @"$REQUEST_JSON"
echo
