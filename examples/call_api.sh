#!/usr/bin/env bash
# 调用漏检复核 API 的示例。用法：
#   ./examples/call_api.sh [BASE_URL] [REQUEST_JSON] [ENDPOINT]
# 默认 BASE_URL=http://localhost:8000，REQUEST_JSON=examples/request_solved.json，
# ENDPOINT=/api/v1/review。归因示例：
#   ./examples/call_api.sh http://localhost:8000 examples/request_attribution.json /api/v1/review/attribution
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
REQUEST_JSON="${2:-examples/request_solved.json}"
ENDPOINT="${3:-/api/v1/review}"

echo "== GET  /health =="
curl -sS "$BASE_URL/health"
echo
echo "== POST ${ENDPOINT}  ($REQUEST_JSON) =="
curl -sS -X POST "$BASE_URL${ENDPOINT}" \
  -H 'Content-Type: application/json' \
  -d @"$REQUEST_JSON"
echo
