"""
pdf_preprocessor.py
===================

Módulo de preprocesamiento de PDF para la Etapa 1 de CODEFEST AD ASTRA 2026
(construcción de la base de conocimiento vectorial).

Uso básico
----------
    from pdf_preprocessor import process_pdf

    jsonl = process_pdf("ruta/al/documento.pdf", fenomeno=1)
    # 'jsonl' es una cadena en formato JSON Lines: un chunk por línea.

    # Para guardarlo en disco:
    with open("documento.jsonl", "w", encoding="utf-8") as f:
        f.write(jsonl)

La clase PDFPreprocessor encapsula todo el flujo de las Secciones 2 y 3
del documento guía:
    - Extracción de texto por página (con OCR de respaldo automático).
    - Limpieza y normalización.
    - Detección de idioma.
    - Asignación de doc_id.
    - Chunking respetando 250 palabras, 250 tokens y completitud lingüística.

Dependencias:
    pip install pymupdf pytesseract pillow langdetect transformers pysbd
    # Binario del sistema (motor OCR):
    #   Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-por
"""

import io
import re
import json
import gzip  # noqa: F401  (reservado por si se procesan PDFs comprimidos)
import hashlib
import unicodedata

import pymupdf                       # PyMuPDF (motor de extracción de PDF)
import pytesseract                   # OCR clásico (Tesseract)
from PIL import Image
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer
import pysbd


# Detección de idioma determinista (mismo resultado en cada corrida)
DetectorFactory.seed = 0


class PDFPreprocessor:
    """
    Preprocesa un archivo PDF y produce chunks con el contrato de salida
    exigido por la Tabla 1 del documento guía.

    Parámetros del constructor
    ---------------------------
    encoder_name : str
        Nombre del encoder de HuggingFace cuyo tokenizer se usa para
        contar tokens. DEBE ser el mismo encoder con el que luego se
        generen los embeddings (Sección 4.3). Por defecto es un modelo
        multilingüe de ejemplo; cámbialo por el encoder oficial del equipo.
    max_palabras : int
        Límite máximo de palabras por chunk (por defecto 250).
    max_tokens : int
        Límite máximo de tokens por chunk (por defecto 250).
    ocr_idiomas : str
        Idiomas que Tesseract intentará reconocer (por defecto es+en+pt).
    ocr_dpi : int
        Resolución a la que se rasteriza una página para OCR.
    min_caracteres_texto : int
        Si una página tiene menos de estos caracteres de texto real, se
        considera escaneada/solo-imagen y se le aplica OCR de respaldo.
    """

    # Frases típicas de plantilla (boilerplate) sin valor informativo.
    FRASES_BOILERPLATE = [
        "suscríbete", "subscribe", "compartir en", "share on",
        "todos los derechos reservados", "all rights reserved",
        "política de cookies", "cookie policy",
        "leer más", "read more",
    ]

    def __init__(
        self,
        encoder_name: str = "intfloat/multilingual-e5-base",
        max_palabras: int = 250,
        max_tokens: int = 250,
        ocr_idiomas: str = "spa+eng+por",
        ocr_dpi: int = 200,
        min_caracteres_texto: int = 20,
    ):
        self.max_palabras = max_palabras
        self.max_tokens = max_tokens
        self.ocr_idiomas = ocr_idiomas
        self.ocr_dpi = ocr_dpi
        self.min_caracteres_texto = min_caracteres_texto

        # El tokenizer se carga una sola vez al crear el objeto.
        self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _contar_tokens(self, texto: str) -> int:
        """Número de tokens según el tokenizer del encoder,
        sin contar tokens especiales ([CLS], [SEP], etc.)."""
        return len(self._tokenizer.encode(texto, add_special_tokens=False))

    @staticmethod
    def _asignar_doc_id(fuente: str) -> str:
        """doc_id único y estable derivado de la fuente (misma fuente,
        mismo doc_id en cada corrida)."""
        hash_corto = hashlib.sha1(fuente.encode("utf-8")).hexdigest()[:8]
        return f"DOC-{hash_corto}"

    @staticmethod
    def _limpiar_texto(texto: str) -> str:
        """Normaliza a UTF-8 (NFC), elimina caracteres de control
        invisibles y colapsa espacios/saltos de línea redundantes."""
        texto = unicodedata.normalize("NFC", texto)
        texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
        lineas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
        return "\n".join(lineas)

    def _eliminar_boilerplate(self, texto: str) -> str:
        """Descarta líneas que coincidan con frases de plantilla."""
        filtradas = []
        for linea in texto.splitlines():
            low = linea.lower()
            if any(frase in low for frase in self.FRASES_BOILERPLATE):
                continue
            filtradas.append(linea)
        return "\n".join(filtradas)

    @staticmethod
    def _detectar_idioma(texto: str) -> str:
        """Detecta el idioma predominante. Devuelve 'desconocido' si el
        texto es muy corto o si la detección falla."""
        limpio = texto.strip()
        if len(limpio.split()) < 5:
            return "desconocido"
        try:
            return detect(limpio)
        except Exception:
            return "desconocido"

    # ------------------------------------------------------------------
    # Extracción de texto (con OCR de respaldo)
    # ------------------------------------------------------------------
    def _ocr_pagina(self, pagina) -> str:
        """Aplica OCR a una página sin texto real: la rasteriza a imagen
        y usa Tesseract (OCR clásico) para transcribir el texto."""
        matriz = pymupdf.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
        pix = pagina.get_pixmap(matrix=matriz)
        imagen = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            return pytesseract.image_to_string(imagen, lang=self.ocr_idiomas)
        except Exception:
            # Si el OCR falla (idioma no instalado, etc.), no romper el flujo.
            return ""

    def _extraer_texto(self, ruta_pdf: str) -> str:
        """Extrae el texto del PDF página por página, preservando el orden
        de lectura. Aplica OCR de respaldo en páginas escaneadas/solo-imagen.

        La detección de "página sin texto" combina dos señales:
          - Muy pocos caracteres extraídos, y
          - Ausencia de bloques de texto en la página.
        """
        documento = pymupdf.open(ruta_pdf)
        paginas = []

        for pagina in documento:
            texto = pagina.get_text().strip()

            # Señal robusta de página escaneada: casi sin texto real.
            sin_texto = (
                len(texto) < self.min_caracteres_texto
                or not pagina.get_text("blocks")
            )
            if sin_texto:
                texto = self._ocr_pagina(pagina).strip()

            if texto:
                paginas.append(texto)

        documento.close()
        # Doble salto de línea como frontera entre páginas.
        return "\n\n".join(paginas)

    @staticmethod
    def _extraer_metadata(ruta_pdf: str) -> dict:
        """Extrae título y fecha de las propiedades internas del PDF,
        si están presentes. Devuelve cadenas vacías cuando faltan."""
        documento = pymupdf.open(ruta_pdf)
        meta = documento.metadata or {}
        documento.close()

        titulo = (meta.get("title") or "").strip()
        fecha = (meta.get("creationDate") or "").strip()
        # La fecha en PDF suele venir como 'D:20260115120000'; se deja AAAAMMDD.
        if fecha.startswith("D:"):
            fecha = fecha[2:10]
        return {"titulo": titulo, "fecha": fecha}

    # ------------------------------------------------------------------
    # Chunking (Sección 3)
    # ------------------------------------------------------------------
    def _partir_oracion_larga(self, oracion: str) -> list[str]:
        """Parte una 'oración' demasiado larga (normalmente un bloque que el
        segmentador no supo dividir) en sub-fragmentos que respeten los
        límites. Primero intenta por fronteras suaves (; : ,); si una parte
        aún excede, hace corte duro por palabras como último recurso.
        Así se evita que el texto quede cortado a media frase al mostrarse."""
        partes = re.split(r"(?<=[;:,])\s+", oracion.strip())

        subfrags, actual, pal, tok = [], [], 0, 0
        for parte in partes:
            n_pal = len(parte.split())
            n_tok = self._contar_tokens(parte)

            # Si una sola cláusula todavía excede, corte duro por palabras.
            if n_pal > self.max_palabras or n_tok > self.max_tokens:
                if actual:
                    subfrags.append(" ".join(actual))
                    actual, pal, tok = [], 0, 0
                palabras = parte.split()
                for i in range(0, len(palabras), self.max_palabras):
                    subfrags.append(" ".join(palabras[i:i + self.max_palabras]))
                continue

            if (pal + n_pal > self.max_palabras) or (tok + n_tok > self.max_tokens):
                subfrags.append(" ".join(actual))
                actual, pal, tok = [], 0, 0

            actual.append(parte)
            pal += n_pal
            tok += n_tok

        if actual:
            subfrags.append(" ".join(actual))

        return subfrags

    def _chunk_texto(self, texto: str, idioma: str) -> list[str]:
        """Divide el texto en fragmentos respetando:
          1. max_palabras   2. max_tokens   3. completitud lingüística.
        Devuelve una lista de cadenas (el texto de cada chunk)."""
        lang = idioma if idioma in ("es", "en", "pt") else "es"
        segmentador = pysbd.Segmenter(language=lang, clean=False)
        oraciones = [o.strip() for o in segmentador.segment(texto) if o.strip()]

        chunks = []
        actual, pal, tok = [], 0, 0

        for oracion in oraciones:
            n_pal = len(oracion.split())
            n_tok = self._contar_tokens(oracion)

            # Caso borde: una oración sola ya excede algún límite. Esto suele
            # ocurrir cuando el segmentador falla con texto ruidoso de PDF y
            # trata un bloque grande como una sola "oración". En vez de
            # emitirla (y que quede cortada al mostrarse), se parte por
            # fronteras suaves (; : ,) y, como último recurso, por palabras.
            if n_pal > self.max_palabras or n_tok > self.max_tokens:
                if actual:
                    chunks.append(" ".join(actual))
                    actual, pal, tok = [], 0, 0
                chunks.extend(self._partir_oracion_larga(oracion))
                continue

            # Si añadir la oración excede palabras O tokens, cerrar el chunk.
            if (pal + n_pal > self.max_palabras) or (tok + n_tok > self.max_tokens):
                chunks.append(" ".join(actual))
                actual, pal, tok = [], 0, 0

            actual.append(oracion)
            pal += n_pal
            tok += n_tok

        if actual:
            chunks.append(" ".join(actual))

        return chunks

    # ------------------------------------------------------------------
    # API pública de la clase
    # ------------------------------------------------------------------
    def procesar(self, ruta_pdf: str, fenomeno: int,
                 fuente: str | None = None, formato: str = "pdf") -> list[dict]:
        """Procesa un PDF completo y devuelve la lista de chunks (dicts)
        con todos los campos del contrato de salida (Tabla 1)."""
        if fuente is None:
            fuente = ruta_pdf  # por defecto, la fuente es la ruta del archivo

        meta = self._extraer_metadata(ruta_pdf)
        cuerpo = self._extraer_texto(ruta_pdf)
        cuerpo = self._limpiar_texto(cuerpo)
        cuerpo = self._eliminar_boilerplate(cuerpo)
        idioma = self._detectar_idioma(cuerpo)
        doc_id = self._asignar_doc_id(fuente)

        chunks_texto = self._chunk_texto(cuerpo, idioma)

        registros = []
        for posicion, texto_chunk in enumerate(chunks_texto):
            registros.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-chunk-{posicion:03d}",
                "fuente": fuente,
                "formato": formato,
                "fenomeno": fenomeno,
                "posicion": posicion,
                "num_tokens": self._contar_tokens(texto_chunk),
                "num_palabras": len(texto_chunk.split()),
                "texto": texto_chunk,
                "idioma": idioma,
                "fecha": meta["fecha"],
                "titulo": meta["titulo"],
            })
        return registros

    def procesar_a_jsonl(self, ruta_pdf: str, fenomeno: int,
                         fuente: str | None = None, formato: str = "pdf") -> str:
        """Igual que procesar(), pero devuelve una cadena en formato
        JSON Lines: un objeto JSON por línea, sin separadores extra."""
        registros = self.procesar(ruta_pdf, fenomeno, fuente, formato)
        return "\n".join(
            json.dumps(r, ensure_ascii=False) for r in registros
        )


# ----------------------------------------------------------------------
# Función de conveniencia solicitada: process_pdf
# ----------------------------------------------------------------------
def process_pdf(pdf_path: str, fenomeno: int,
                encoder_name: str = "intfloat/multilingual-e5-base") -> str:
    """
    Procesa un archivo PDF y devuelve el JSONL con los chunks del archivo.

    Parámetros
    ----------
    pdf_path : str
        Ruta al archivo PDF a procesar.
    fenomeno : int
        Fenómeno temático al que pertenece el documento (1, 2 o 3).
    encoder_name : str, opcional
        Encoder de HuggingFace cuyo tokenizer se usa para contar tokens.
        Debe coincidir con el encoder usado para generar los embeddings.

    Retorna
    -------
    str
        Cadena en formato JSON Lines (un chunk por línea). Cada línea es
        un objeto JSON con los campos de la Tabla 1 del documento guía.
    """
    preprocesador = PDFPreprocessor(encoder_name=encoder_name)
    return preprocesador.procesar_a_jsonl(pdf_path, fenomeno)


if __name__ == "__main__":
    # Pequeña demostración de uso (requiere un PDF real en la ruta dada).
    import sys
    if len(sys.argv) >= 3:
        ruta = sys.argv[1]
        fen = int(sys.argv[2])
        print(process_pdf(ruta, fen))
    else:
        print("Uso: python pdf_preprocessor.py <ruta_pdf> <fenomeno>")
