# Base de Conocimiento — CODEFEST AD ASTRA 2026 (Etapa 1)

Pipeline para construir una **base de conocimiento vectorial** a partir de un corpus multi-formato (PDF, CSV, JSON, imágenes, PBF), siguiendo las restricciones técnicas del reto: modelos _encoder_ (familia BERT / XLM-RoBERTa), indexación con FAISS y similitud coseno.

## Tabla de contenido

- [Descripción](#descripción)
- [Arquitectura del pipeline](#arquitectura-del-pipeline)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Datos](#datos)
- [Documentación del proyecto](#documentación-del-proyecto)
- [Estado del proyecto](#estado-del-proyecto)
- [Equipo](#equipo)
- [Licencia](#licencia)

## Descripción

Sistema para la construcción de una base de conocimiento a partir del procesamiento, segmentación y representación vectorial de diferentes tipos de documentos.

El proyecto permite transformar documentos de entrada en fragmentos de texto estructurados (chunks), generar representaciones vectoriales mediante un modelo de embeddings y construir una base vectorial que posteriormente puede utilizarse para realizar búsquedas semánticas.

## Arquitectura del pipeline

El flujo general de procesamiento es:

```
Corpus (PDF / CSV / JSON / imágenes / PBF)
        │
        ▼
Preprocesamiento por formato (scripts/preprocessing/*_preprocessor.py, image_reader.py)
        │
        ▼
Limpieza, chunking y detección de idioma (text_processor.py, language_detection.py)
        │
        ▼
Codificación semántica con modelos encoder (encode_db.py)
        │
        ▼
Indexación vectorial FAISS (IndexFlatIP, similitud coseno) + metadata.jsonl
        │
        ▼
Consulta → Generación de resultados (scripts/preprocessing/entrega/generador.py)
```

Restricciones técnicas respetadas por el pipeline:

- Solo se usan modelos **encoder** para indexación y recuperación (no decoder).
- Similitud coseno mediante `IndexFlatIP` con vectores normalizados.
- El orden de inserción en FAISS coincide con el orden de chunks volcado a `metadata.jsonl`.

## Estructura del repositorio

## Requisitos

- Python `>3.8`
- Dependencias del pipeline de preprocesamiento (`scripts/preprocessing/requirements.txt`):
  - `torch` (CPU), `pymupdf`, `pytesseract`, `pillow` — extracción de PDF/OCR
  - `mapbox-vector-tile` — extracción de PBF
  - `langdetect`, `pysbd`, `transformers`, `pandas`, `openpyxl` — procesamiento de texto/tabular
  - `faiss-cpu`, `sentence-transformers` — base vectorial
  - `jupyter`, `ipywidgets` — notebooks
- **Tesseract OCR** instalado a nivel de sistema (requerido por `pytesseract`).

## Instalación

```bash
git clone https://github.com/JOliverosRIng/base_de_conocimiento.git
cd base_de_conocimiento

python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# Paquete del proyecto
pip install -e .

# Dependencias del pipeline de preprocesamiento
pip install -r scripts/preprocessing/requirements.txt
```

## Datos

- `Inputs/Indice_Datos_Codefest.xlsx`: índice de las fuentes de datos del reto.
- `Inputs/Extracto_Preguntas_50_v2.pdf`: extracto de preguntas de referencia para evaluar la base de conocimiento.
- El corpus completo (`CORPUS CODEFEST AD ASTRA 2026`).

## Equipo

- [Javier Alejandro Penagos Hernández]
- [Leonardo Castañeda]
- [Nicolas Felipe Corredor Cortes]
- [Janeth Oliveros Ramirez]
