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
        invisibles, quita marcadores de página ruidosos (p. ej. '!96 !')
        que rompen el segmentador de oraciones, y colapsa espacios/saltos
        de línea redundantes."""
        texto = unicodedata.normalize("NFC", texto)
        texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)

        # Quitar marcadores de página tipo '!96 !', '! 100 !', '!101'
        texto = re.sub(r"!\s*\d+\s*!?", " ", texto)
        # NUEVO: quitar encabezados repetidos '! AI INDEX, NOVEMBER 2017'
        texto = re.sub(
            r"!\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9,.\s]{3,60}?\d{4}", " ", texto
        )
        # Colapsar secuencias de signos '!' sueltos
        texto = re.sub(r"\s*!\s*!\s*", " ", texto)
        texto = re.sub(r"\s+!\s+", " ", texto)   # NUEVO

        texto = re.sub(r"[ \t]{2,}", " ", texto)
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
        """Parte un bloque que el segmentador entregó como una sola 'oración'
        pero que excede los límites. Estrategia, de más a menos preferible:
          1. Re-segmentar por puntos/fin de oración que pysbd no detectó.
          2. Si una oración real aún excede, partir por fronteras suaves
             (; :) y luego comas, agrupando SIN superar el límite.
          3. Solo si una sola cláusula sin puntuación interna sigue siendo
             más grande que el límite, corte duro por palabras (último
             recurso inevitable).
        Nunca deja una unidad partida entre dos fragmentos salvo el caso 3."""
        # Paso 1: reintentar separar oraciones por signos de fin de oración
        # (punto, interrogación, exclamación) seguidos de espacio y mayúscula.
        unidades = re.split(r"(?<=[.?])\s+(?=[A-ZÁÉÍÓÚÑ])", oracion.strip())
        unidades = [u.strip() for u in unidades if u.strip()]

        subfrags, actual, pal, tok = [], [], 0, 0
        for unidad in unidades:
            n_pal = len(unidad.split())
            n_tok = self._contar_tokens(unidad)

            # Si esta unidad todavía excede, intentar dividirla por cláusulas.
            if n_pal > self.max_palabras or n_tok > self.max_tokens:
                if actual:
                    subfrags.append(" ".join(actual))
                    actual, pal, tok = [], 0, 0
                subfrags.extend(self._partir_por_clausulas(unidad))
                continue

            # Cerrar el fragmento en curso si la unidad no cabe entera.
            if (pal + n_pal > self.max_palabras) or (tok + n_tok > self.max_tokens):
                subfrags.append(" ".join(actual))
                actual, pal, tok = [], 0, 0

            actual.append(unidad)
            pal += n_pal
            tok += n_tok

        if actual:
            subfrags.append(" ".join(actual))

        return subfrags

    def _partir_por_clausulas(self, texto: str) -> list[str]:
        """Divide un texto que no tiene fin de oración claro usando fronteras
        suaves (; :) y comas. Como último recurso, corte por palabras. No
        fuerza el llenado: cierra en cuanto la siguiente cláusula no cabe."""
        partes = re.split(r"(?<=[;:,])\s+", texto.strip())

        subfrags, actual, pal, tok = [], [], 0, 0
        for parte in partes:
            n_pal = len(parte.split())
            n_tok = self._contar_tokens(parte)

            # Cláusula sin puntuación interna que aún excede: corte por palabras.
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
    

    def _chunk_texto(self, texto: str, idioma: str,
                     max_palabras_ventana: int = 250,
                     solape_palabras: int = 80) -> list[str]:
        """Divide el texto con ventana deslizante y solape:
          1. Cada ventana tiene como máximo 'max_palabras_ventana' palabras
             (250) — puede ser MENOR para no cortar oraciones.
          2. También respeta el techo de tokens (max_tokens).
          3. Completitud lingüística: la ventana solo agrupa oraciones
             completas; nunca parte una frase.
          4. Cada chunk se solapa con el siguiente hasta 'solape_palabras'
             (80) — puede ser MENOR, retrocediendo por oraciones completas.
        Devuelve una lista de cadenas (el texto de cada chunk)."""
        lang = idioma if idioma in ("es", "en", "pt") else "es"
        segmentador = pysbd.Segmenter(language=lang, clean=False)
        oraciones = [o.strip() for o in segmentador.segment(texto) if o.strip()]
 
        # Pre-cálculo de palabras/tokens por oración.
        info = [(o, len(o.split()), self._contar_tokens(o)) for o in oraciones]
        n = len(info)
 
        # El solape no puede ser mayor o igual que la ventana.
        solape_palabras = min(solape_palabras, max_palabras_ventana - 1)
 
        chunks = []
        i = 0
        while i < n:
            actual, pal, tok = [], 0, 0
            j = i
 
            while j < n:
                oracion, n_pal, n_tok = info[j]
 
                # Oración individual que excede el techo de la ventana o de
                # tokens: se parte aparte (respeta cláusulas, no la mete cruda).
                if n_pal > max_palabras_ventana or n_tok > self.max_tokens:
                    if actual:
                        break  # primero cerramos lo acumulado
                    chunks.extend(self._partir_oracion_larga(oracion))
                    j += 1
                    i = j
                    actual = None  # marca: bloque ya emitido
                    break
 
                # Si la siguiente oración no cabe, cerrar la ventana aquí.
                if (pal + n_pal > max_palabras_ventana) or (tok + n_tok > self.max_tokens):
                    break
 
                actual.append(oracion)
                pal += n_pal
                tok += n_tok
                j += 1
 
            # Si el bloque se emitió por partición, continuar sin solape.
            if actual is None:
                continue
 
            if actual:
                chunks.append(" ".join(actual))
 
            if j >= n:
                break
 
            # --- Retroceso para el solape (por oraciones completas) ---
            solape_acum = 0
            inicio_siguiente = j
            k = j - 1
            while k > i:
                _, kp, _ = info[k]
                if solape_acum + kp > solape_palabras:
                    break
                solape_acum += kp
                inicio_siguiente = k
                k -= 1
 
            # Garantía de avance: nunca quedarse en el mismo inicio.
            i = inicio_siguiente if inicio_siguiente > i else j
 
        return chunks
    

    # ------------------------------------------------------------------
    # Posprocesamiento de chunks (recorte de bordes cortados)
    # ------------------------------------------------------------------
    def _posprocesar_chunk(self, texto: str) -> str:
        """Recorta los bordes de un chunk SOLO si están cortados:

        Regla 1 (inicio): si el texto empieza en minúscula (viene de media
            oración), elimina todo hasta la primera secuencia '. Mayúscula'
            y conserva desde esa mayúscula. Si no hay '. Mayúscula', usa la
            primera ',' o ';' como punto de corte (sin exigir mayúscula).

        Regla 2 (final): si el texto NO termina en '.', '!' o '?' (oración
            incompleta), elimina la última oración que empieza tras un
            '. Mayúscula'. Si no hay '. Mayúscula', usa la última ',' o ';'
            como punto de corte.

        Si el borde ya está sano, no se toca. Devuelve el texto recortado."""
        texto = texto.strip()
        if not texto:
            return texto

        # --- Regla 1: recortar inicio solo si empieza cortado (minúscula) ---
        if texto[0].islower():
            # Buscar la primera secuencia '. Mayúscula'.
            m = re.search(r"\.\s+(?=[A-ZÁÉÍÓÚÑ])", texto)
            if m:
                texto = texto[m.end():].lstrip()
            else:
                # Respaldo: primera ',' o ';'.
                m2 = re.search(r"[;,]\s+", texto)
                if m2:
                    texto = texto[m2.end():].lstrip()

        # --- Regla 2: recortar final solo si termina cortado ---
        if texto and texto[-1] not in ".!?":
            # Buscar el último '. Mayúscula' y cortar ahí (conservando el '.').
            matches = list(re.finditer(r"\.\s+(?=[A-ZÁÉÍÓÚÑ])", texto))
            if matches:
                ultimo = matches[-1]
                texto = texto[:ultimo.start() + 1].rstrip()
            else:
                # Respaldo: última ',' o ';'.
                matches2 = list(re.finditer(r"[;,]", texto))
                if matches2:
                    ultimo2 = matches2[-1]
                    texto = texto[:ultimo2.start() + 1].rstrip()

        return texto.strip()

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
            texto_chunk = self._posprocesar_chunk(texto_chunk)
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
# Función de salida: process_pdf
# ----------------------------------------------------------------------
def process_pdf(pdf_path: str, fenomeno: int,
                fuente: str | None = None, encoder_name: str = "intfloat/multilingual-e5-base") -> str:
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
    return preprocesador.procesar_a_jsonl(pdf_path, fenomeno, fuente)


if __name__ == "__main__":
    # Pequeña demostración de uso (requiere un PDF real en la ruta dada).
    import sys
    if len(sys.argv) > 3:
        ruta = sys.argv[1]
        fen = int(sys.argv[2])
        fuente = sys.argv[3] 
        print(process_pdf(ruta, fen,fuente))
    else:
        print("Uso: python pdf_preprocessor.py <ruta_pdf> <fenomeno> <fuente>")
