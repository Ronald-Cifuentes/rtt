#!/usr/bin/env bash
#
# Setup script for RTT (Real-Time Speech-to-Speech Translation).
# Run from the repository root: ./setup.sh
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=============================================="
echo "  RTT — Setup"
echo "=============================================="
echo ""

# ── Python ──
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found. Install Python 3.10+ and try again."
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "📌 Python: $PY_VERSION"

# ── Backend venv & deps ──
echo ""
echo "📦 Backend: virtualenv and dependencies..."
cd "$REPO_ROOT/backend"
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "   Created backend/venv"
fi
# shellcheck source=/dev/null
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "   ✅ Backend dependencies installed"
cd "$REPO_ROOT"

# ── .env ──
if [ ! -f ".env" ]; then
  cp env.example .env
  echo ""
  echo "📄 Created .env from env.example (edit .env to change device, models, etc.)"
else
  echo ""
  echo "📄 .env already exists (unchanged)"
fi

# ── Frontend ──
echo ""
echo "📦 Frontend: npm install..."
cd "$REPO_ROOT/frontend"
if command -v npm &>/dev/null; then
  npm install
  echo "   ✅ Frontend dependencies installed"
else
  echo "   ⚠️  npm not found — skip frontend. Install Node.js 18+ and run: cd frontend && npm install"
fi
cd "$REPO_ROOT"

# ── Sanity check (optional) ──
echo ""
echo "🔍 Sanity check (backend)..."
export PYTHONPATH="$REPO_ROOT"
if "$REPO_ROOT/backend/venv/bin/python" scripts/sanity_check.py 2>/dev/null; then
  echo ""
else
  echo "   ⚠️  Some checks failed (e.g. qwen-asr). Run: cd backend && source venv/bin/activate && pip install -r requirements.txt"
fi

echo ""
echo "=============================================="
echo "  Setup complete"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. (Optional) Pre-download models:"
echo "     PYTHONPATH=. backend/venv/bin/python scripts/download_models.py"
echo ""
echo "  2. Start backend:"
echo "     cd backend && source venv/bin/activate && python -m app.main"
echo ""
echo "  3. In another terminal, start frontend:"
echo "     cd frontend && npm run dev"
echo ""
echo "  4. Open http://localhost:5173"
echo ""
