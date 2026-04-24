#!/usr/bin/env bash
# Start DocuCluster: FastAPI backend + Vite frontend, then open browser.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cleanup() {
  echo ""
  echo "[start] stopping…"
  jobs -p | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[start] backend → http://localhost:8000"
( cd backend && uvicorn main:app --reload --port 8000 ) &

echo "[start] frontend → http://localhost:5173"
( cd frontend && npm run dev ) &

sleep 3

URL="http://localhost:5173"
case "$(uname -s)" in
  Darwin) open "$URL" ;;
  Linux)  xdg-open "$URL" >/dev/null 2>&1 || true ;;
  MINGW*|MSYS*|CYGWIN*) start "" "$URL" ;;
  *) echo "[start] open $URL in your browser" ;;
esac

wait
