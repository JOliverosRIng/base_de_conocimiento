# Base de Conocimiento — CODEFEST AD ASTRA 2026 (Etapa 1)

Pipeline para construir una **base de conocimiento vectorial** a partir de un corpus multi-formato (PDF, CSV, JSON, imágenes, PBF), siguiendo las restricciones técnicas del reto: modelos _encoder_ (familia BERT / XLM-RoBERTa), indexación con FAISS y similitud coseno.

## Tabla de contenido

- [Descripción](#descripción)
- [Arquitectura del pipeline](#arquitectura-del-pipeline)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Datos](#datos)
- [Equipo](#equipo)

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

La siguiente estructura refleja la organización actual del repositorio y los principales artefactos del proyecto:

```text
base_de_conocimiento/
├── README.md
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── Inputs/
│   ├── Indice_Datos_Codefest.xlsx
│   └── Extracto_Preguntas_50_v2.pdf
├── docs/
│   ├── acceptance/
│   │   └── exit_report.md
│   ├── business_understanding/
│   │   └── project_charter.md
│   ├── data/
│   │   ├── data_definition.md
│   │   ├── data_dictionary.md
│   │   └── data_summary.md
│   ├── deployment/
│   │   └── deploymentdoc.md
│   └── modeling/
│       ├── baseline_models.md
│       └── model_report.md
├── output/                      # generado, ignorado en git
│   ├── changelog.jsonl
│   ├── metadata.json
│   └── resumen.json
├── output_bk/                   # generado, ignorado en git
│   ├── changelog.jsonl
│   └── resumen.json
├── scripts/
│   └── preprocessing/
│       ├── main.py
│       ├── csv_preprocessor.py
│       ├── json_preprocessor.py
│       ├── pdf_preprocessor.py
│       ├── pbf_preprocessor.py
│       ├── text_preprocessor.py
│       ├── text_processor.py
│       ├── language_detection.py
│       ├── image_reader.py
│       ├── pdf_reader.py
│       ├── requirements.txt
│       ├── orquestador.ipynb
│       ├── embeddings_gpu.ipynb
│       ├── recover_not_processed.ipynb
│       ├── ejecuciones/
│       │   └── orquestador.ipynb
│       ├── tests/
│       │   └── test_chunk_coverage.py
│       └── entrega/
│           ├── generador.py
│           ├── consulta_jsonl
│           ├── resultados.jsonl
│           ├── informe tecnico
│           └── base_vectorial/
│               └── encoder_multilingual/
│                   ├── index.faiss
│                   └── metadata.jsonl
└── src/
    ├── CORPUS CODEFEST AD ASTRA 2026/   # corpus del reto, ignorado en git
    │   ├── F1_IA_y_Capacidades_Estrategicas/
    │   │   ├── AI_Index_Stanford/
    │   │   ├── Atlantic_Council/
    │   │   ├── CENIA/
    │   │   ├── CSET_Georgetown/
    │   │   ├── DAIO/
    │   │   ├── Defensa21_LatAm/
    │   │   ├── ILIA_Latam/
    │   │   └── RutaN_GEIAL/
    │   ├── F2_Seguridad_Entorno_Espacial/
    │   │   ├── CSIS_Aerospace/
    │   │   ├── ESA_Space_Debris/
    │   │   ├── INPE/
    │   │   ├── SWF_Counterspace/
    │   │   └── UNOOSA/
│   └── F3_Dinamicas_Territoriales/
│       ├── Alertas_Tempranas/
│       ├── Amazon_Underworld/
│       ├── CEEEP/
│       ├── CEOBS/
│       ├── MAPP_OEA/
│       ├── RESDAL/
│       ├── SIPRI/
│       └── Wilson_Center/

```


Este repositorio está organizado en torno a tres grandes bloques:

- `docs/`: documentación del proyecto, entregables y reportes de análisis, modelado y despliegue.
- `scripts/preprocessing/`: componentes de ETL, limpieza, chunking, OCR, procesamiento por formato y exportación de resultados.
- `src/`: código fuente del paquete Python y corpus base del reto, incluyendo los datos temáticos por temática del proyecto.
- `output/` y `output_bk/`: resultados generados por el pipeline y copias de seguridad de artefactos.
- `Inputs/`: archivos de entrada y referencia para la competencia.

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
