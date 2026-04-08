#!/usr/bin/env bash
# validate-submission.sh — see hackathon docs; expects openenv-core + Docker + curl.
set -uo pipefail
PING_URL="${1:-}"
REPO_DIR="${2:-.}"
if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi
REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)" || exit 1
PING_URL="${PING_URL%/}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30) || HTTP_CODE="000"
echo "HF /reset HTTP: $HTTP_CODE"
if [ ! -f "$REPO_DIR/Dockerfile" ] && [ ! -f "$REPO_DIR/server/Dockerfile" ]; then
  echo "No Dockerfile at repo root or server/"
  exit 1
fi
CTX="$REPO_DIR"
[ -f "$REPO_DIR/Dockerfile" ] || CTX="$REPO_DIR/server"
docker build "$CTX" || exit 1
( cd "$REPO_DIR" && openenv validate ) || exit 1
echo "All checks passed."
exit 0
