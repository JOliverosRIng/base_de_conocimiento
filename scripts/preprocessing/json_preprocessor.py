import json
from pathlib import Path
import pysbd
from transformers import AutoTokenizer
from langdetect import detect
import hashlib


class Chunker:

    def __init__(self, max_words=250):
        self.max_words = max_words

        self.tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-base"
        )

    def split(self, document: dict, fenomeno: int):

        texto = document["content"]
        metadata = document["metadata"]

        idioma = metadata.get("language", "en")

        try:
            segmentador = pysbd.Segmenter(
                language=idioma,
                clean=False
            )
        except Exception:
            segmentador = pysbd.Segmenter(
                language="en",
                clean=False
            )

        oraciones = segmentador.segment(texto)

        chunks = []
        chunk_actual = []
        palabras_actuales = 0
        posicion = 0

        doc_id = metadata["doc_id"]

        for oracion in oraciones:

            oracion = oracion.strip()

            if not oracion:
                continue

            palabras_oracion = len(oracion.split())

            if palabras_actuales + palabras_oracion <= self.max_words:

                chunk_actual.append(oracion)
                palabras_actuales += palabras_oracion

            else:

                if chunk_actual:

                    texto_chunk = " ".join(chunk_actual)

                    num_words = len(texto_chunk.split())

                    num_tokens = len(
                        self.tokenizer.encode(
                            texto_chunk,
                            add_special_tokens=False
                        )
                    )

                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}-chunk-{posicion:03d}",
                        "fuente": metadata["source"],
                        "formato": metadata["format"],
                        "fenomeno": fenomeno,
                        "posicion": posicion,
                        "num_tokens": num_tokens,
                        "num_words": num_words,
                        "texto": texto_chunk,
                        "idioma": metadata.get("language"),
                        "fecha": metadata.get("date"),
                        "titulo": metadata.get("title")
                    })

                    posicion += 1

                chunk_actual = [oracion]
                palabras_actuales = palabras_oracion

        if chunk_actual:

            texto_chunk = " ".join(chunk_actual)

            num_words = len(texto_chunk.split())

            num_tokens = len(
                self.tokenizer.encode(
                    texto_chunk,
                    add_special_tokens=False
                )
            )

            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-chunk-{posicion:03d}",
                "fuente": metadata["source"],
                "formato": metadata["format"],
                "fenomeno": fenomeno,
                "posicion": posicion,
                "num_tokens": num_tokens,
                "num_words": num_words,
                "texto": texto_chunk,
                "idioma": metadata.get("language"),
                "fecha": metadata.get("date"),
                "titulo": metadata.get("title")
            })

        return chunks


class JSONExtractor:

    def _ruta_relativa(self, ruta: str) -> str:  #Este se lo copie a Leo
        CARPETA_RAIZ = "CORPUS CODEFEST AD ASTRA 2026"
        ruta_norm = ruta.replace("\\", "/")
        idx = ruta_norm.find(CARPETA_RAIZ)
        if idx != -1:
            return ruta_norm[idx:]
        return ruta_norm

    def _asignar_doc_id(self, fuente: str) -> str: #Este se lo copie a leo
        hash_corto = hashlib.sha1(
            fuente.encode("utf-8")
        ).hexdigest()[:8]
        return f"DOC-{hash_corto}"

    def _detectar_idioma(self, texto: str) -> str: #Este tambien se lo copie a leo
        limpio = texto.strip()
        if len(limpio.split()) < 5:
            return "desconocido"
        try:
            return detect(limpio)
        except Exception:
            return "desconocido"
    
    def _aplanar_json(self, valor) -> str:  #Se agregó esta función para que si el json es completamente diferente como es el caso del swf_counterspace_2026 o el de catalog
                                            #Se aplane y se agregue toda esta info al texto del chunk
        partes = []

        if isinstance(valor, dict):

            for clave, contenido in valor.items():

                if contenido is None:
                    continue

                texto_contenido = self._aplanar_json(contenido)

                if texto_contenido:
                    partes.append(
                        f"{clave}: {texto_contenido}"
                    )

        elif isinstance(valor, list):

            for elemento in valor:

                texto_elemento = self._aplanar_json(elemento)

                if texto_elemento:
                    partes.append(texto_elemento)

        else:

            partes.append(str(valor))

        return "\n".join(partes)


    

    def extract(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):

            body_text = data.get("body_text")

            if not body_text:
                body_text = self._aplanar_json(data)

            title = data.get("title")
            date = data.get("date")

        elif isinstance(data, list):

            body_text = self._aplanar_json(data)

            title = None
            date = None

        else:

            body_text = str(data)

            title = None
            date = None

        body_text = body_text.strip()
        idioma = self._detectar_idioma(body_text)
        ruta_relativa = self._ruta_relativa(file_path)
        doc_id = self._asignar_doc_id(ruta_relativa)
        metadata = {
            "doc_id": doc_id,
            "source": Path(file_path).name,
            "format": "json",
            "title": title,
            "date": date,
            "language": idioma
        }

        return {
            "content": body_text,
            "metadata": metadata
        }


def procesar_json(path: str, fenomeno: int) -> list[dict]:

    extractor = JSONExtractor()

    chunker = Chunker(
        max_words=250
    )

    documento = extractor.extract(path)

    chunks = chunker.split(
        documento,
        fenomeno
    )

    return chunks

def procesar_jsonl(path: str, fenomeno: int) -> str: #Esta es la función a llamar desde afuera
    chunks = procesar_json(path, fenomeno)

    return "\n".join(
        json.dumps(chunk, ensure_ascii=False)
        for chunk in chunks
    )


