"""
pbf_preprocessor.py
h ===================
"""

import re
import json
import gzip
import hashlib
import unicodedata
import mapbox_vector_tile

from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer
import pysbd


# Detección de idioma determinista (mismo resultado en cada corrida)
DetectorFactory.seed = 0


class PBFPreprocessor:

    def __init__(
        self,
        encoder_name: str = "intfloat/multilingual-e5-base",
        max_palabras: int = 250,
        max_tokens: int = 250,
        separador_pares: str = " | ",
        debug_huge_token: bool = False,
    ):
        self.max_palabras = max_palabras
        self.max_tokens = max_tokens
        self.separador_pares = separador_pares
        self.debug_huge_token = debug_huge_token
        # El tokenizer se carga una sola vez al crear el objeto.
        self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    @staticmethod
    def _ruta_relativa(ruta: str) -> str:
        """Devuelve la ruta relativa desde la carpeta 'CORPUS CODEFEST AD
        ASTRA 2026' en adelante, con separadores normalizados a '/'. Esto
        hace que el doc_id y la sean reproducibles entre máquinas
        (Windows/Linux) sin depender de dónde esté el proyecto.

        Si la ruta no contiene esa carpeta, se usa el path absoluto
        normalizado a '/' como respaldo."""
        CARPETA_RAIZ = "CORPUS CODEFEST AD ASTRA 2026"
        # Normalizar separadores a '/' (Windows usa '\', Linux usa '/').
        ruta_norm = ruta.replace("\\", "/")

        idx = ruta_norm.find(CARPETA_RAIZ)
        if idx != -1:
            return ruta_norm[idx:]
        # Respaldo: path absoluto normalizado.
        return ruta_norm

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
        """Normaliza a UTF-8 (NFC), elimina caracteres de control invisibles
        y colapsa espacios/saltos de línea redundantes."""
        texto = unicodedata.normalize("NFC", texto)
        texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
        texto = re.sub(r"[ \t]{2,}", " ", texto)
        lineas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
        return "\n".join(lineas)

    @staticmethod
    def _detectar_idioma(texto: str) -> str:
        """Detecta el idioma predominante. Devuelve 'desconocido' si el
        texto es muy corto o si la detección falla."""
        limpio = texto.strip() # divide por palabras y descarta espacios iniciales y finales
        if len(limpio.split()) < 5: # si hay menos de 5 palabras, no se puede detectar el idioma
            return "desconocido"
        try:
            return detect(limpio)
        except Exception:
            return "desconocido"

    # ------------------------------------------------------------------
    # Decodificación y extracción de texto del PBF
    # ------------------------------------------------------------------
    @staticmethod
    def _leer_bytes(ruta_pbf: str) -> bytes:
        """Lee el PBF crudo; si viene comprimido con gzip, lo descomprime."""
        with open(ruta_pbf, "rb") as f: #Abre este archivo (r:read; b:binary) y lo asigna a la variable f
            datos = f.read() # Lee los datos del archivo
        # Bytes mágicos de gzip: 0x1f 0x8b
        if datos[:2] == b"\x1f\x8b":
            datos = gzip.decompress(datos)
        return datos

    def _extraer_texto(self, ruta_pbf: str) -> str:
        """Decodifica la tesela con mapbox_vector_tile y convierte los
        atributos de cada feature del mapa en texto ('atributo: valor'),
        recorriendo todas las capas. Descarta features duplicadas (mismo
        conjunto de atributos repetido dentro del archivo).
        """
        datos = self._leer_bytes(ruta_pbf)
        tesela = mapbox_vector_tile.decode(datos)

        lineas = []
        vistos = set()

        for nombre_capa, capa in tesela.items():
            for feature in capa.get("features", []):
                propiedades = feature.get("properties", {})
                if not propiedades:
                    continue

                pares = [
                    f"{clave}: {valor}"
                    for clave, valor in propiedades.items()
                    if valor not in (None, "")
                ]
                if not pares:
                    continue

                texto_elemento = self.separador_pares.join(pares)

                # Deduplicación dentro del archivo.
                if texto_elemento in vistos:
                    continue
                vistos.add(texto_elemento)

                # El nombre de la capa aporta contexto temático.
                lineas.append(f"[{nombre_capa}] {texto_elemento}")

        return "\n".join(lineas)

    # ------------------------------------------------------------------
    # Chunking (idéntico al módulo PDF: ventana deslizante + solape)
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
                bloque = []
                bloque_pal = 0
                bloque_tok = 0

                for palabra in palabras:
                    palabra_tok = self._contar_tokens(palabra)

                    if (
                        bloque
                        and (
                            bloque_pal + 1 > self.max_palabras
                            or bloque_tok + palabra_tok > self.max_tokens
                         )
                    ):
                        subfrags.append(" ".join(bloque))
                        bloque = []
                        bloque_pal = 0
                        bloque_tok = 0

                    bloque.append(palabra)
                    bloque_pal += 1
                    bloque_tok += palabra_tok

                if bloque:
                    subfrags.append(" ".join(bloque))
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
        #lang = idioma if idioma in ("es", "en", "pt") else "es"
        # pysbd NO soporta portugués ('pt'); se usa español como respaldo,
        # que comparte las reglas de puntuación para segmentar oraciones.
        lang = idioma if idioma in ("es", "en") else "es"
        segmentador = pysbd.Segmenter(language=lang, clean=False)
        oraciones = [o.strip() for o in segmentador.segment(texto) if o.strip()]

        # Pre-cálculo de palabras/tokens por oración.
        info = []

        for o in oraciones:
            n_tok = self._contar_tokens(o)

            if self.debug_huge_token and n_tok > 512:
                print("=" * 70)
                print(
                    f"[DEBUG_HUGE_TOKEN] Bloque de {n_tok} tokens "
                    f"({len(o.split())} palabras):"
                )
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
    # API pública de la clase
    # ------------------------------------------------------------------
    def procesar(
        self,
        ruta_pbf: str,
        fenomeno: int,
        fuente: str | None = None,
        formato: str = "pbf"
    ) -> list[dict]:

        fuente = self._ruta_relativa(ruta_pbf)

        cuerpo = self._extraer_texto(ruta_pbf)
        cuerpo = self._limpiar_texto(cuerpo)

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
                "fecha": "",
                "titulo": "",
            })

        return registros

    def procesar_a_jsonl(self, ruta_pbf: str, fenomeno: int,
                         fuente: str | None = None, formato: str = "pbf") -> str:
        """Igual que procesar(), pero devuelve una cadena JSON Lines:
        un objeto JSON por línea, sin separadores extra."""
        registros = self.procesar(ruta_pbf, fenomeno, fuente, formato)
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in registros)


# ----------------------------------------------------------------------
# Función de salida: process_pbf
# ----------------------------------------------------------------------

def process_pbf(
    pbf_path: str,
    fenomeno: int,
    fuente: str | None = None,
    encoder_name: str = "intfloat/multilingual-e5-base",
    debug_huge_token: bool = False
) -> str:
    """
    Procesa un archivo PBF y devuelve el JSONL con los chunks del archivo.

    Parámetros
    ----------
    pbf_path : str
        Ruta al archivo PBF a procesar.

    fenomeno : int
        Fenómeno temático al que pertenece el documento (1, 2 o 3).

    fuente : str, opcional
        Fuente del documento. Si no se proporciona, se utiliza la
        ruta relativa del archivo.

    encoder_name : str, opcional
        Encoder de HuggingFace cuyo tokenizer se usa para contar tokens.
        Debe coincidir con el encoder usado para generar los embeddings.

    debug_huge_token : bool, opcional
        Si es True, imprime los bloques que superan 512 tokens para
        diagnóstico.

    Retorna
    -------
    str
        Cadena en formato JSON Lines (un chunk por línea). Cada línea es
        un objeto JSON con los campos de la Tabla 1 del documento guía.
    """

    preprocesador = PBFPreprocessor(
        encoder_name=encoder_name,
        debug_huge_token=debug_huge_token
    )

    return preprocesador.procesar_a_jsonl(
        pbf_path,
        fenomeno,
        fuente
    )