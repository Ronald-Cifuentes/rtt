# Análisis Crítico de Errores ASR - Diagnóstico y Correcciones (histórico)

> **Nota:** Este documento describe problemas con el ASR anterior (Faster-Whisper). El proyecto actual usa **Qwen3-ASR**. Se conserva como referencia histórica.

## 🔴 PROBLEMA IDENTIFICADO

### Síntoma
Transcripción completamente incorrecta de texto en español:
- **Original**: "A lo largo de la historia y desde la invención de la escritura, han sido múltiples los ejemplos de autores que a través de esta han dado rienda suelta a su imaginación con el fin de expresar sus sentimientos, emociones y pensamientos. Muchos de ellos han plasmado diferentes creencias, valores y maneras de hacer o vivir, algunos incluso en un corto espacio."

- **Transcrito**: "a lo largo de la historia y es... desde la inmensión de la escritura ha sido muy un ejemplo de autores. Nación, con el fin de expresar sus sentimientos, emociones. pensamientos. Muchos de ellos son plazas por diferentes prenses de boludo. en el corte espasi."

- **Traducido**: "throughout history and it's... since the immension of writing has been an example of authors. Nation, in order to express their feelings, emotions. Thoughts. Many of them are squares for different prenses of boludo. in the spasi cut."

## 🔍 ANÁLISIS DE CAUSA RAÍZ

### 1. PROBLEMA PRINCIPAL: Modelo Whisper "base" INADECUADO ⚠️

**Causa**: El modelo Whisper "base" tiene solo **74 millones de parámetros**, insuficiente para transcripción precisa de español.

### 2. PROBLEMAS SECUNDARIOS

- Parámetros ASR subóptimos (beam_size, VAD)
- Ventana de contexto corta
- La traducción funcionaba correctamente; el error estaba en el ASR

## ✅ CORRECCIONES APLICADAS (en su momento)

- Cambio de modelo a small/medium
- Mejora de parámetros (beam_size, best_of, VAD)
- Aumento de WINDOW_SEC a 8.0
- Filtros de confianza y de alucinaciones

**Estado actual:** El ASR fue reemplazado por **Qwen3-ASR**; ver `docs/ANALISIS_QWEN3_ASR_MIGRACION.md`.

---

**Fecha de análisis**: 2024 | **Estado**: Histórico (ASR actual = Qwen3-ASR)
