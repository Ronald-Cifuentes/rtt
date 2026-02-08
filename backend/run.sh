#!/usr/bin/env bash
# Run backend from repo root: ./backend/run.sh
# Or from backend: ./run.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
  echo "Creating venv and installing dependencies..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Ensure qwen-asr is installed (in case venv was created before it was in requirements)
pip install -q 'qwen-asr>=0.0.6'

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
