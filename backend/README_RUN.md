# Cómo arrancar el backend (ASR = Qwen3-ASR)

**Setup rápido:** Desde la raíz del repo ejecuta `./setup.sh` (crea venv, instala deps, copia .env).

**Importante:** Ejecuta desde este repositorio; en otras copias del proyecto puede estar el ASR antiguo (faster-whisper).

## Desde la raíz del repo (qyf)

```bash
# 1. Ir a la raíz del repo (donde están backend/ y scripts/)
cd /Users/ronaldcifuentes/.cursor/worktrees/rtt/qyf

# 2. Crear venv e instalar dependencias del backend (incluye qwen-asr)
python3.10 -m venv venv
source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# 3. (Opcional) Comprobar que todo está instalado
PYTHONPATH=backend python scripts/sanity_check.py

# 4. (Opcional) Pre-descargar el modelo ASR
PYTHONPATH=backend python scripts/download_models.py --asr-model Qwen/Qwen3-ASR-0.6B

# 5. Arrancar el servidor (desde la raíz, para que encuentre app y .env)
cd backend
python -m app.main
# O desde la raíz: PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Al arrancar deberías ver en log:
```text
ASR loaded: Qwen3-ASR Qwen/Qwen3-ASR-0.6B on cpu (...)
```
Si ves `faster-whisper` es que estás en otro directorio / otro `requirements.txt`.

## Si ya tienes un venv dentro de backend

```bash
cd /Users/ronaldcifuentes/.cursor/worktrees/rtt/qyf/backend
source venv/bin/activate
pip install -r requirements.txt   # debe instalar qwen-asr, NO faster-whisper
python -m app.main
```

Comprueba que `requirements.txt` contiene `qwen-asr>=0.0.6` y **no** `faster-whisper`.
