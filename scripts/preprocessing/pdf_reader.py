"""
pdf_reader.py
=============

Lectura de PDF para CODEFEST AD ASTRA 2026. Se encarga ÚNICAMENTE de
extraer el texto crudo del PDF (con OCR de respaldo para páginas
escaneadas) y su metadata. Es específico de la fuente PDF.

El procesamiento del texto (limpieza, chunking, validación) NO vive aquí:
está en text_processor.py, que es reutilizable para cualquier fuente.

Uso básico
----------
    from pdf_reader import PDFReader

    reader = PDFReader()
    resultado = reader.extraer("documento.pdf")
    # resultado = {"texto": "...", "titulo": "...", "fecha": "..."}

Dependencias:
    pip install pymupdf pytesseract pillow
    # Binario del sistema (motor OCR):
    #   Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-por
"""

import io

import pymupdf                 # PyMuPDF (motor de extracción de PDF)
import pytesseract             # OCR clásico (Tesseract)
from PIL import Image


class PDFReader:
    """
    Lee un PDF y devuelve su texto crudo en renglones, más su metadata
    (título, fecha). Aplica OCR de respaldo en páginas sin texto real.

    Parámetros del constructor
    ---------------------------
    ocr_idiomas : str
        Idiomas que Tesseract intentará reconocer (por defecto es+en+pt).
    ocr_dpi : int
        Resolución a la que se rasteriza una página para OCR.
    min_caracteres_texto : int
        Si una página tiene menos de estos caracteres de texto real, se
        considera escaneada/solo-imagen y se le aplica OCR de respaldo.
    """

    def __init__(
        self,
        ocr_idiomas: str = "spa+eng+por",
        ocr_dpi: int = 200,
        min_caracteres_texto: int = 20,
    ):
        self.ocr_idiomas = ocr_idiomas
        self.ocr_dpi = ocr_dpi
        self.min_caracteres_texto = min_caracteres_texto

    # ------------------------------------------------------------------
    # OCR de respaldo
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

    # ------------------------------------------------------------------
    # Extracción de texto y metadata
    # ------------------------------------------------------------------
    def _extraer_texto(self, ruta_pdf: str) -> str:
        """Extrae el texto del PDF página por página, preservando el orden
        de lectura, y lo devuelve como un string con saltos de línea entre
        renglones. Aplica OCR de respaldo en páginas escaneadas/solo-imagen.

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
    # API pública
    # ------------------------------------------------------------------
    def extraer(self, ruta_pdf: str) -> dict:
        """Lee el PDF y devuelve el texto crudo (string con saltos de línea
        entre renglones) más metadata.

        Retorna
        -------
        dict
            {"texto": str, "titulo": str, "fecha": str}
        """
        texto = self._extraer_texto(ruta_pdf)
        meta = self._extraer_metadata(ruta_pdf)
        return {
            "texto": texto,
            "titulo": meta["titulo"],
            "fecha": meta["fecha"],
        }
