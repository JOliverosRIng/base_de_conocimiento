"""
text_processor.py
=================

Procesamiento de texto REUTILIZABLE para CODEFEST AD ASTRA 2026,
independiente de la fuente. Recibe texto crudo en renglones (de un PDF,
JSON, PBF, etc.) y produce los chunks con el contrato de salida (Tabla 1).

Contiene:
  - TextProcessor: limpieza, chunking (ventana + solape), posprocesamiento
    de bordes, detección de idioma, conteo de tokens, doc_id, validación.
  - process_pdf: función orquestadora que une PDFReader (lectura) con
    TextProcessor (procesamiento) para producir el JSONL de un PDF.

Uso básico
----------
    from text_processor import process_pdf

    jsonl = process_pdf("documento.pdf", fenomeno=1)

    # O usando el procesador directamente con texto de cualquier fuente:
    from text_processor import TextProcessor
    tp = TextProcessor()
    chunks = tp.procesar_texto(texto, fuente="ruta/rel", fenomeno=1)

Dependencias:
    pip install langdetect transformers pysbd
"""

import re
import json
import hashlib
import unicodedata

from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer
import pysbd


# Detección de idioma determinista (mismo resultado en cada corrida)
DetectorFactory.seed = 0


class TextProcessor:
    """
    Procesa texto crudo (renglones) y produce chunks con el contrato de
    salida (Tabla 1). Es independiente de la fuente: sirve para PDF, JSON,
    PBF, etc.

    Parámetros del constructor
    ---------------------------
    encoder_name : str
        Encoder de HuggingFace cuyo tokenizer se usa para contar tokens.
        DEBE coincidir con el encoder usado para generar los embeddings.
    max_palabras : int
        Techo de palabras por chunk (por defecto 250).
    max_tokens : int
        Techo de tokens por chunk (por defecto 250).
    debug_huge_token : bool
        Si es True, imprime los bloques que superan 512 tokens (los que
        disparan la advertencia del tokenizer) para diagnóstico.
    imprimir_validacion : bool
        La validación SIEMPRE se ejecuta y se guarda en
        self.resultados_validacion_integridad. Esta bandera solo controla
        si se IMPRIMEN los problemas (imprime si está activa Y hay problemas).
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
        debug_huge_token: bool = False,
        imprimir_validacion: bool = False,
    ):
        self.max_palabras = max_palabras
        self.max_tokens = max_tokens
        self.debug_huge_token = debug_huge_token
        self.imprimir_validacion = imprimir_validacion
        self.resultados_validacion_integridad = None

        # El tokenizer se carga una sola vez. Primero desde caché local
        # (evita contactar a HuggingFace); solo descarga si no está.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                encoder_name, local_files_only=True
            )
        except Exception:
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
        """doc_id único y estable derivado de la fuente."""
        hash_corto = hashlib.sha1(fuente.encode("utf-8")).hexdigest()[:8]
        return f"DOC-{hash_corto}"

    @staticmethod
    def _ruta_relativa(ruta: str) -> str:
        """Devuelve la ruta relativa desde 'CORPUS CODEFEST AD ASTRA 2026'
        en adelante, con separadores normalizados a '/'. Reproducible entre
        máquinas. Si no contiene esa carpeta, usa el path absoluto a '/'."""
        CARPETA_RAIZ = "CORPUS CODEFEST AD ASTRA 2026"
        ruta_norm = ruta.replace("\\", "/")
        idx = ruta_norm.find(CARPETA_RAIZ)
        if idx != -1:
            return ruta_norm[idx:]
        return ruta_norm

    @staticmethod
    def _limpiar_texto(texto: str) -> str:
        """Limpieza completa y autocontenida. Recibe el texto crudo (string
        con saltos de línea) y devuelve el texto ya como párrafo único (una
        sola línea, espacios sencillos):
          - Normaliza a UTF-8 (NFC) y quita caracteres de control.
          - Remueve comillas dobles (rectas y curvas); conserva simples.
          - Quita marcadores de página ruidosos ('!96 !', encabezados).
          - Procesa renglón por renglón: colapsa espacios y, tras ello,
            descarta los renglones de boilerplate (frases de plantilla).
          - Elimina puntos redundantes ('. . . .' de índices/rellenos).
          - Une los renglones con un solo espacio (párrafo único)."""
        texto = unicodedata.normalize("NFC", texto)
        texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)

        # Remover solo comillas DOBLES (rectas " y curvas “ ”). Todas las
        # comillas simples y apóstrofos (rectos ' y curvos ‘ ’) se conservan.
        texto = re.sub(r"[\"\u201c\u201d]", "", texto)

        # Quitar marcadores de página tipo '!96 !', '! 100 !', '!101'
        texto = re.sub(r"!\s*\d+\s*!?", " ", texto)
        # Quitar encabezados repetidos '! AI INDEX, NOVEMBER 2017'
        texto = re.sub(
            r"!\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9,.\s]{3,60}?\d{4}", " ", texto
        )
        # Colapsar secuencias de signos '!' sueltos
        texto = re.sub(r"\s*!\s*!\s*", " ", texto)
        texto = re.sub(r"\s+!\s+", " ", texto)

        # Procesar renglón por renglón: colapsar espacios y descartar
        # los renglones que sean boilerplate.
        lineas_limpias = []
        for linea in texto.splitlines():
            linea = re.sub(r"[ \t]{2,}", " ", linea).strip()
            if not linea:
                continue
            low = linea.lower()
            if any(frase in low for frase in TextProcessor.FRASES_BOILERPLATE):
                continue
            lineas_limpias.append(linea)

        # Unir en un párrafo único con un solo espacio.
        texto = " ".join(lineas_limpias)

        # Eliminar puntos redundantes ('. . . .' separados por un espacio):
        # reemplazar por un espacio y colapsar dobles para no dejar dobles.
        texto = re.sub(r"(?:\.\s){1,}\.", " ", texto)
        texto = re.sub(r"[ \t]{2,}", " ", texto)

        return texto.strip()

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
    # Chunking (Sección 3)
    # ------------------------------------------------------------------
    def _partir_oracion_larga(self, oracion: str) -> list[str]:
        """Parte un bloque que el segmentador entregó como una sola 'oración'
        pero que excede los límites. Estrategia, de más a menos preferible:
          1. Re-segmentar por puntos/fin de oración que pysbd no detectó.
          2. Si una oración real aún excede, partir por fronteras suaves
             (; :) y luego comas, agrupando SIN superar el límite.
          3. Solo si una sola cláusula sin puntuación interna sigue siendo
             más grande que el límite, corte duro por palabras.
        Nunca deja una unidad partida entre dos fragmentos salvo el caso 3."""
        unidades = re.split(r"(?<=[.?])\s+(?=[A-ZÁÉÍÓÚÑ])", oracion.strip())
        unidades = [u.strip() for u in unidades if u.strip()]

        subfrags, actual, pal, tok = [], [], 0, 0
        for unidad in unidades:
            n_pal = len(unidad.split())
            n_tok = self._contar_tokens(unidad)

            if n_pal > self.max_palabras or n_tok > self.max_tokens:
                if actual:
                    subfrags.append(" ".join(actual))
                    actual, pal, tok = [], 0, 0
                subfrags.extend(self._partir_por_clausulas(unidad))
                continue

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
        """Divide un texto sin fin de oración claro usando fronteras suaves
        (; :) y comas. Como último recurso, corte por palabras. No fuerza el
        llenado: cierra en cuanto la siguiente cláusula no cabe."""
        partes = re.split(r"(?<=[;:,])\s+", texto.strip())

        subfrags, actual, pal, tok = [], [], 0, 0
        for parte in partes:
            n_pal = len(parte.split())
            n_tok = self._contar_tokens(parte)

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
          1. Ventana de máximo 'max_palabras_ventana' palabras (puede ser menor).
          2. Respeta el techo de tokens (max_tokens).
          3. Completitud lingüística: solo agrupa oraciones completas.
          4. Solape de hasta 'solape_palabras' (puede ser menor), retrocediendo
             por oraciones completas."""
        lang = idioma if idioma in ("es", "en", "pt") else "es"
        segmentador = pysbd.Segmenter(language=lang, clean=False)
        oraciones = [o.strip() for o in segmentador.segment(texto) if o.strip()]

        # Pre-cálculo de palabras/tokens por oración.
        info = []
        for o in oraciones:
            n_tok = self._contar_tokens(o)
            if self.debug_huge_token and n_tok > 512:
                print("=" * 70)
                print(f"[DEBUG_HUGE_TOKEN] Bloque de {n_tok} tokens "
                      f"({len(o.split())} palabras):")
                print(o)
                print("=" * 70)
            info.append((o, len(o.split()), n_tok))
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

                if n_pal > max_palabras_ventana or n_tok > self.max_tokens:
                    if actual:
                        break
                    chunks.extend(self._partir_oracion_larga(oracion))
                    j += 1
                    i = j
                    actual = None
                    break

                if (pal + n_pal > max_palabras_ventana) or (tok + n_tok > self.max_tokens):
                    break

                actual.append(oracion)
                pal += n_pal
                tok += n_tok
                j += 1

            if actual is None:
                continue

            if actual:
                chunks.append(" ".join(actual))

            if j >= n:
                break

            # Retroceso para el solape (por oraciones completas).
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

            i = inicio_siguiente if inicio_siguiente > i else j

        return chunks

    # ------------------------------------------------------------------
    # Posprocesamiento de chunks (recorte de bordes cortados)
    # ------------------------------------------------------------------
    def _posprocesar_chunk(self, texto: str, texto_anterior: str | None = None) -> str:
        """Recorta los bordes de un chunk SOLO si están cortados (Reglas 1 y 2).
        La Regla 1b usa el chunk anterior (solape) para detectar fragmentos
        cortados al inicio. Si el borde ya está sano, no se toca."""
        texto = texto.strip()
        if not texto:
            return texto

        # --- Regla 1a: recortar inicio si empieza cortado (minúscula) ---
        if texto[0].islower():
            m = re.search(r"\.\s+(?=[A-ZÁÉÍÓÚÑ])", texto)
            if m:
                texto = texto[m.end():].lstrip()
            else:
                m2 = re.search(r"[;,]\s+", texto)
                if m2:
                    texto = texto[m2.end():].lstrip()

        # --- Regla 1b: lógica del solape con el chunk anterior ---
        if texto_anterior:
            m_punto = re.search(r"(?<!\d)\.\s+(?=[A-ZÁÉÍÓÚÑ])", texto)
            if m_punto:
                candidata = texto[:m_punto.end()].strip()
                pos = texto_anterior.find(candidata)
                if pos != -1:
                    previo = texto_anterior[:pos].rstrip()
                    if previo and not previo.endswith("."):
                        texto = texto[m_punto.end():].lstrip()

        # --- Regla 2: recortar final solo si termina cortado ---
        if texto and texto[-1] not in ".!?":
            matches = list(re.finditer(r"\.\s+(?=[A-ZÁÉÍÓÚÑ])", texto))
            if matches:
                ultimo = matches[-1]
                texto = texto[:ultimo.start() + 1].rstrip()
            else:
                matches2 = list(re.finditer(r"[;,]", texto))
                if matches2:
                    ultimo2 = matches2[-1]
                    texto = texto[:ultimo2.start() + 1].rstrip()

        return texto.strip()

    # ------------------------------------------------------------------
    # Validación de integridad de los chunks
    # ------------------------------------------------------------------
    def _validar_chunks(self, cuerpo: str, chunks: list[dict],
                        n_ancla: int = 8, imprimir: bool = False) -> dict:
        """Valida la integridad de los chunks contra el cuerpo original.
        SIEMPRE devuelve un dict con los problemas. Imprime solo si
        'imprimir' es True Y hay problemas.

        Validación 1 (continuidad): ubica cada chunk por sus primeras
            'n_ancla' palabras y verifica inicio(k+1) < fin(k).
        Validación 2 (límites): ningún chunk supera max_palabras/max_tokens."""
        palabras_orig = cuerpo.split()
        n_orig = len(palabras_orig)

        def ubicar(texto_chunk, desde=0):
            pal = texto_chunk.split()
            if not pal:
                return (None, None)
            ancla = pal[:n_ancla]
            for i in range(desde, n_orig - len(ancla) + 1):
                if palabras_orig[i:i + len(ancla)] == ancla:
                    return (i, i + len(pal))
            return (None, None)

        # Validación 1: continuidad por posiciones.
        posiciones = []
        no_ubicados = []
        cursor = 0
        for idx, c in enumerate(chunks):
            ini, fin = ubicar(c["texto"], desde=cursor)
            if ini is None:
                ini, fin = ubicar(c["texto"], desde=0)
            if ini is None:
                no_ubicados.append(idx)
            else:
                posiciones.append((idx, ini, fin))
                cursor = ini

        rupturas = []
        for a in range(len(posiciones) - 1):
            idx_k, ini_k, fin_k = posiciones[a]
            idx_k1, ini_k1, fin_k1 = posiciones[a + 1]
            if ini_k1 >= fin_k:
                contexto = " ".join(palabras_orig[fin_k:ini_k1][:20])
                rupturas.append({
                    "entre_chunk": idx_k,
                    "y_chunk": idx_k1,
                    "palabras_perdidas_aprox": ini_k1 - fin_k,
                    "texto_perdido_inicio": contexto,
                })

        # Validación 2: límites de tokens y palabras.
        excede_palabras = []
        excede_tokens = []
        for idx, c in enumerate(chunks):
            n_pal = len(c["texto"].split())
            n_tok = self._contar_tokens(c["texto"])
            if n_pal > self.max_palabras:
                excede_palabras.append((idx, n_pal))
            if n_tok > self.max_tokens:
                excede_tokens.append((idx, n_tok))

        problemas = {
            "rupturas": rupturas,
            "chunks_no_ubicados": no_ubicados,
            "exceden_palabras": excede_palabras,
            "exceden_tokens": excede_tokens,
        }
        hay_problemas = bool(
            rupturas or no_ubicados or excede_palabras or excede_tokens
        )
        problemas["hay_problemas"] = hay_problemas

        if imprimir and hay_problemas:
            print("=" * 60)
            print("⚠️  VALIDACIÓN DE INTEGRIDAD — problemas encontrados:")
            if rupturas:
                print(f"  Rupturas (texto perdido): {len(rupturas)}")
                for r in rupturas:
                    print(f"    · Entre chunk {r['entre_chunk']} y "
                          f"{r['y_chunk']}: ~{r['palabras_perdidas_aprox']} "
                          f"palabras. Empieza en: ...{r['texto_perdido_inicio']}...")
            if no_ubicados:
                print(f"  Chunks no ubicados en el original: {no_ubicados}")
            if excede_palabras:
                print(f"  Exceden {self.max_palabras} palabras: {excede_palabras}")
            if excede_tokens:
                print(f"  Exceden {self.max_tokens} tokens: {excede_tokens}")
            print("=" * 60)

        return problemas

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def procesar_texto(self, texto: str, fuente: str, fenomeno: int,
                       formato: str = "pdf", titulo: str = "",
                       fecha: str = "") -> list[dict]:
        """Procesa texto crudo (string) y devuelve la lista de chunks
        (dicts) con todos los campos del contrato de salida (Tabla 1).

        Es independiente de la fuente: el texto puede venir de PDF, JSON,
        PBF, etc. La 'fuente' se normaliza a ruta relativa desde
        'CORPUS CODEFEST AD ASTRA 2026' para reproducibilidad.
        """
        fuente = self._ruta_relativa(fuente)

        cuerpo = self._limpiar_texto(texto)
        idioma = self._detectar_idioma(cuerpo)
        doc_id = self._asignar_doc_id(fuente)

        chunks_texto = self._chunk_texto(cuerpo, idioma)

        registros = []
        for posicion, texto_chunk in enumerate(chunks_texto):
            anterior = chunks_texto[posicion - 1] if posicion > 0 else None
            texto_chunk = self._posprocesar_chunk(texto_chunk, anterior)
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
                "fecha": fecha,
                "titulo": titulo,
            })

        # La validación SIEMPRE se ejecuta y se guarda. La bandera solo
        # controla si se imprimen los problemas.
        self.resultados_validacion_integridad = self._validar_chunks(
            cuerpo, registros, imprimir=self.imprimir_validacion
        )

        return registros

    def procesar_a_jsonl(self, texto: str, fuente: str, fenomeno: int,
                        formato: str = "pdf", titulo: str = "",
                        fecha: str = "") -> str:
        """Igual que procesar_texto(), pero devuelve una cadena JSON Lines."""
        registros = self.procesar_texto(
            texto, fuente, fenomeno, formato, titulo, fecha
        )
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in registros)


# ----------------------------------------------------------------------
# Función orquestadora: process_pdf (une PDFReader + TextProcessor)
# ----------------------------------------------------------------------
def process_pdf(pdf_path: str, fenomeno: int,
                encoder_name: str = "intfloat/multilingual-e5-base",
                debug_huge_token: bool = False,
                imprimir_validacion: bool = False) -> str:
    """
    Procesa un PDF y devuelve el JSONL con los chunks del archivo.
    Orquesta la lectura (PDFReader) y el procesamiento (TextProcessor).

    Parámetros
    ----------
    pdf_path : str
        Ruta al archivo PDF a procesar.
    fenomeno : int
        Fenómeno temático (1, 2 o 3).
    encoder_name : str, opcional
        Encoder cuyo tokenizer se usa para contar tokens.
    debug_huge_token : bool, opcional
        Imprime bloques que superan 512 tokens (diagnóstico).
    imprimir_validacion : bool, opcional
        Imprime los problemas de validación si los hay.

    Retorna
    -------
    str
        JSON Lines (un chunk por línea) con los campos de la Tabla 1.
    """
    # Import local para no forzar la dependencia de PyMuPDF si solo se usa
    # el TextProcessor con otras fuentes.
    from pdf_reader import PDFReader

    reader = PDFReader()
    datos = reader.extraer(pdf_path)

    procesador = TextProcessor(
        encoder_name=encoder_name,
        debug_huge_token=debug_huge_token,
        imprimir_validacion=imprimir_validacion,
    )
    return procesador.procesar_a_jsonl(
        texto=datos["texto"],
        fuente=pdf_path,
        fenomeno=fenomeno,
        formato="pdf",
        titulo=datos["titulo"],
        fecha=datos["fecha"],
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        print(process_pdf(sys.argv[1], int(sys.argv[2])))
    else:
        print("Uso: python text_processor.py <ruta_pdf> <fenomeno>")
