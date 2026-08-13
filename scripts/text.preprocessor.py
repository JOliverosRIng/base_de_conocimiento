import hashlib
from pathlib import Path
import json
import pysbd
from transformers import AutoTokenizer
from langdetect import detect

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
                        "fuente": metadata["fuente"],
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
                "fuente": metadata["fuente"],
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
    

class TXTExtractor:

    def _ruta_relativa(self, ruta: str) -> str:
        CARPETA_RAIZ = "CORPUS CODEFEST AD ASTRA 2026"

        ruta_norm = ruta.replace("\\", "/")
        idx = ruta_norm.find(CARPETA_RAIZ)

        if idx != -1:
            return ruta_norm[idx:]

        return ruta_norm

    def _asignar_doc_id(self, fuente: str) -> str:
        hash_corto = hashlib.sha1(
            fuente.encode("utf-8")
        ).hexdigest()[:8]

        return f"DOC-{hash_corto}"

    def _detectar_idioma(self, texto: str) -> str:
        limpio = texto.strip()

        if len(limpio.split()) < 5:
            return "desconocido"

        try:
            return detect(limpio)
        except Exception:
            return "desconocido"

    def extract(self, file_path: str) -> dict:

        with open(file_path, "r", encoding="utf-8") as file:
            texto = file.read()

        texto = texto.strip()

        ruta_relativa = self._ruta_relativa(file_path)

        doc_id = self._asignar_doc_id(ruta_relativa)

        idioma = self._detectar_idioma(texto)

        metadata = {
            "doc_id": doc_id,
            "source": Path(file_path).name,
            "format": "txt",
            "title": None,
            "date": None,
            "language": idioma,
            "fuente": ruta_relativa
        }

        return {
            "content": texto,
            "metadata": metadata
        }

def procesar_txt(path: str, fenomeno: int) -> list[dict]:

    extractor = TXTExtractor()

    chunker = Chunker(
        max_words=250
    )

    documento = extractor.extract(path)

    chunks = chunker.split(
        documento,
        fenomeno
    )

    return chunks

def procesar_txtl(path: str, fenomeno: int) -> str:

    chunks = procesar_txt(path, fenomeno)

    return "\n".join(
        json.dumps(chunk, ensure_ascii=False)
        for chunk in chunks
    )


