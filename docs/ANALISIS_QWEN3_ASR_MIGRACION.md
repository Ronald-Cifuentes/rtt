# Análisis en profundidad: código actual vs Qwen3-ASR y viabilidad de migración

## 1. Resumen del código actual (ASR)

### 1.1 Stack actual
- **Motor**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 + Whisper).
- **Modelo**: Whisper por tamaño (`tiny` | `base` | `small` | `medium` | `large-v2` | `large-v3`), configurable vía `ASR_MODEL_SIZE` (default `small`).
- **Modo**: Pseudo-streaming con ventana deslizante:
  1. Cada `ASR_INTERVAL_MS` (500 ms) se toma el último `WINDOW_SEC` (8 s) del `AudioBuffer`.
  2. Se transcribe esa ventana con Whisper.
  3. Se devuelve el texto completo de la ventana; el `CommitTracker` decide qué parte “comprometer”.

### 1.2 Contrato del ASR actual
- **Entrada**: `audio: np.ndarray` (float32, 16 kHz), `language: str` (ej. `"es"`, `"en"`).
- **Salida**: `str` (texto transcrito) o `""` si se omite por silencio/ruido/filtros.
- **Uso**: El orquestador llama `await self.asr.transcribe(audio, language=self.source_lang)`; el idioma viene del mensaje `config` del WebSocket (`source_lang`).

### 1.3 Defensas contra alucinaciones (actuales)
- **Puerta de energía**: RMS &lt; `_MIN_RMS_ENERGY` → no se llama al modelo.
- **VAD**: Silero integrado en faster-whisper (`vad_filter=True`, `vad_parameters`).
- **Por segmento**: se descartan segmentos con `no_speech_prob` &gt; umbral o `avg_logprob` bajo.
- **Patrones**: regex para “subscribe”, “music”, “gracias por ver”, etc.
- **Repetición**: `_is_repetitive()` para evitar texto repetido/hallucinado.

### 1.4 Dependencias y entorno
- `faster-whisper>=1.0.0`, `torch`, `numpy`.
- Soporte device: `cpu`, `cuda`; en `mps` se usa CPU.
- Sin timestamps en la API pública (solo internos para filtrado).

---

## 2. Qwen3-ASR según documentación e imagen

Referencia: [GitHub Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

### 2.1 Funcionalidades (alineadas con la imagen)
| Funcionalidad | Descripción | En el código actual |
|---------------|-------------|----------------------|
| **Robustez (ruido)** | Reconocimiento estable en entornos ruidosos. | Whisper + VAD + filtros; Qwen3-ASR anuncia “robust recognition under complex acoustic environments”. |
| **Timestamps** | Palabra/frase con tiempos (Qwen3-ForcedAligner-0.6B). | No expuesto; solo texto. |
| **Multilingüe** | 52 idiomas/dialectos + detección automática. | Whisper multilingüe; idioma fijado por cliente (`source_lang`). |
| **Rápido y streaming** | “5h en 10s” en modo async; streaming real. | Pseudo-streaming (ventana fija); no streaming nativo. |
| **Voz cantada / música** | Speech, singing, songs with BGM. | Solo habla; no optimizado para canto. |

### 2.2 Modelos y uso típico
- **Qwen3-ASR-1.7B** y **Qwen3-ASR-0.6B**: ASR + identificación de idioma, 52 idiomas/dialectos.
- **Qwen3-ForcedAligner-0.6B**: alineación texto–audio, timestamps (hasta 5 min, 11 idiomas).
- Backends: **transformers** (mínimo) o **vLLM** (más rápido, streaming).
- Entrada: path local, URL, base64 o `(np.ndarray, sr)`; salida incluye `language` y `text`; opcionalmente `time_stamps` si se usa el aligner.

### 2.3 Ejemplo de API (transformers)
```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="https://.../asr_en.wav",  # o path, base64, (np.ndarray, sr)
    language=None,  # auto, o "English" / "Chinese" para forzar
)
# results[0].language, results[0].text
```
Con timestamps se añade `forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B"` y `return_time_stamps=True`.

---

## 3. Comparativa técnica

### 3.1 Compatibilidad de API
| Aspecto | Actual (faster-whisper) | Qwen3-ASR |
|---------|--------------------------|-----------|
| Entrada audio | `np.ndarray` float32, 16 kHz | Path/URL/base64/`(np.ndarray, sr)` ✅ |
| Idioma | `language="es"` (código) | `language=None` (auto) o nombre "Spanish" ✅ |
| Salida | `str` | `results[0].text` (+ `results[0].language`) ✅ |
| Async | `run_in_executor` sobre llamada síncrona | Llamada bloqueante; habría que seguir usando executor o vLLM async |

El contrato actual “audio + language → string” se puede mantener con un adaptador que:
- Pase el array (y sample rate) en el formato que acepte Qwen3-ASR.
- Convierta `source_lang` (ej. `"es"`) a nombre de idioma que Qwen espere (ej. `"Spanish"`) o `None` para auto.
- Extraiga `results[0].text` y opcionalmente use `results[0].language` para logs o UI.

### 3.2 Idioma
- **Actual**: códigos ISO como `"es"`, `"en"` (desde frontend/WS).
- **Qwen3-ASR**: en ejemplos usan nombres como `"English"`, `"Chinese"`; el repo menciona 52 idiomas/dialectos. Habrá que mantener un mapa `es` → `"Spanish"`, `en` → `"English"`, etc., o usar `language=None` y confiar en la detección automática (útil si en el futuro el cliente no envía idioma).

### 3.3 Streaming real vs pseudo-streaming
- **Actual**: ventana deslizante cada 500 ms; no es streaming de modelo.
- **Qwen3-ASR**: streaming real solo con backend **vLLM**; no con transformers. El README indica que streaming no soporta batch ni timestamps.
- Para no cambiar la arquitectura del orquestador (ventana + CommitTracker), la opción más simple es usar Qwen3-ASR en **modo offline por ventana** (como ahora): cada 500 ms se pasa el último N segundos de audio. Así el resto del pipeline (CommitTracker, MT, TTS) no cambia.
- Una migración futura a streaming real implicaría otro flujo (chunks de audio enviados al modelo en tiempo real y resultados incrementales), lo que afectaría al CommitTracker y posiblemente al protocolo WS.

### 3.4 Timestamps
- **Actual**: no se exponen; solo se usan internamente en segmentos de Whisper para filtrar.
- **Qwen3-ASR**: opcional con ForcedAligner; podría usarse para mejorar commit (p.ej. cortes por tiempo de palabra) o para UI (resaltado por tiempo). No es necesario para reemplazar el ASR actual; se puede añadir en una segunda fase.

### 3.5 Robustez y alucinaciones
- Qwen3-ASR anuncia buena robustez en ruido y “challenging text patterns”.
- Las defensas actuales (RMS, VAD, filtros de segmento, patrones, repetición) son lógica de aplicación; se pueden conservar delante de cualquier motor (incluido Qwen3-ASR) para no perder seguridad.

### 3.6 Rendimiento y recursos
- **Qwen3-ASR-1.7B**: ~1.7B parámetros; mejor calidad, más GPU/RAM.
- **Qwen3-ASR-0.6B**: más ligero; “2000× throughput at concurrency 128” con vLLM.
- **Whisper small**: ~244M; **medium** ~769M. Cambiar a Qwen3-ASR-0.6B/1.7B implica modelos más grandes; vLLM + GPU recomendable para baja latencia.

### 3.7 Dependencias
- **Actual**: `faster-whisper`, `torch`, `numpy`.
- **Qwen3-ASR**: `pip install qwen-asr` (transformers) o `qwen-asr[vllm]`; opcional FlashAttention. Posible conflicto de versiones con `transformers`/`torch` ya usados por TTS u otros; conviene probar en entorno aislado (venv/conda).

---

## 4. Conclusión: ¿tiene Qwen3-ASR “todas las funcionalidades de la imagen” y conviene reemplazar?

### 4.1 Funcionalidades de la imagen
Sí: el repo y la documentación cubren las cinco patas del diagrama:
- Robustez en ruido.
- Timestamps (vía Qwen3-ForcedAligner).
- Multilingüe (52 idiomas/dialectos) y detección de idioma.
- Rápido y streaming (vLLM, async, streaming).
- Voz cantada y canciones con BGM.

Nada de eso está hoy en el pipeline actual de forma explícita (salvo multilingüe limitado al idioma fijado por el cliente).

### 4.2 Conveniencia del reemplazo
- **Ventajas de migrar**:
  - Mejor calidad potencial en español y multilingüe (benchmarks del repo).
  - Opción de timestamps y detección de idioma sin cambiar mucho el contrato (adaptador).
  - Streaming real posible en el futuro (vLLM).
  - Soporte explícito para voz cantada si el producto lo requiere.
- **Inconvenientes / riesgos**:
  - Modelos más pesados (0.6B/1.7B) y dependencias nuevas; recomendable GPU para latencia similar.
  - Mantener dos backends (Whisper vs Qwen) o migrar por completo; testing y despliegue más costosos.
  - Streaming real con vLLM exigiría cambios en el orquestador y en el protocolo si se quiere aprovechar.

**Recomendación**: Es **conveniente** reemplazar el ASR por Qwen3-ASR si se prioriza calidad multilingüe, robustez y posibilidad de timestamps/streaming futuro, y se acepta mayor uso de GPU y un ciclo de integración y pruebas. Para una migración mínima y segura, mantener el mismo patrón (ventana deslizante + mismo contrato `transcribe(audio, language) → str`) con un adaptador sobre Qwen3-ASR (transformers o vLLM en modo no-streaming por ventana).

---

## 5. Pasos sugeridos para la migración

1. **Entorno**: Crear entorno limpio (ej. Python 3.12), instalar `qwen-asr` (y opcionalmente `qwen-asr[vllm]`) junto con las dependencias actuales del backend; resolver conflictos de `torch`/`transformers`.
2. **Adaptador**: Implementar una clase `Qwen3ASREngine` con la misma interfaz que `ASREngine`: `load()`, `async transcribe(audio, language) -> str`. Internamente:
   - Convertir `language` (código) a nombre o `None`.
   - Llamar a `model.transcribe(audio=(audio, 16000), language=...)` en un executor.
   - Devolver `results[0].text` (y opcionalmente usar `results[0].language`).
3. **Config**: Añadir opción (env) para elegir motor: `faster-whisper` vs `qwen3-asr`, y modelo (ej. `Qwen/Qwen3-ASR-0.6B` o `1.7B`).
4. **Defensas**: Reutilizar puerta RMS, filtros de patrón y repetición actuales sobre el texto devuelto por Qwen3-ASR.
5. **Tests**: Probar con los mismos WAV y casos que en `ANALISIS_ERRORES_ASR.md` y comparar calidad y latencia.
6. **Opcional**: Integrar ForcedAligner y exponer timestamps en eventos WS en una fase posterior; o planificar paso a streaming con vLLM cambiando el bucle ASR y el CommitTracker.

---

**Referencias**
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- Código actual: `backend/app/pipeline/asr.py`, `orchestrator.py`, `config.py`, `core/audio_buffer.py`, `core/commit_tracker.py`

---

## 6. Estado de la migración (actualizado)

La migración se realizó: el ASR actual usa **Qwen3-ASR** en lugar de faster-whisper.

- **Motor**: `qwen-asr` con modelo configurable vía `ASR_MODEL` (default `Qwen/Qwen3-ASR-0.6B`).
- **Interfaz**: La misma que antes: `ASREngine.load()` y `await asr.transcribe(audio, language=...)` → `str`.
- **Config**: `ASR_MODEL`, `ASR_MAX_NEW_TOKENS`, `ASR_MAX_BATCH_SIZE` en `config.py` y `env.example`.
- **Dependencias**: `requirements.txt` usa `qwen-asr>=0.0.6`; se eliminó `faster-whisper`.
- **Scripts**: `download_models.py` y `test_wav_pipeline.py` actualizados para Qwen3-ASR; `sanity_check.py` comprueba `qwen_asr` y la config actual.
