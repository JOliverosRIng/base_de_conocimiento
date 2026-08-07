import json
from pathlib import Path
from typing import Any
import pysbd
from transformers import AutoTokenizer


class Chunker:

    def __init__(self, max_words=250, language="es"):
        self.max_words = max_words
        self.segmentador = pysbd.Segmenter(
            language=language,
            clean=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    def split(self, document: dict, doc_id: str):
        texto = document["content"]
        metadata = document["metadata"]
        oraciones = self.segmentador.segment(texto)  # aqui se parte en oraciones
        chunks = []
        chunk_actual = []
        palabras_actuales = 0
        posicion = 0

        for oracion in oraciones: #Aqui se recorren las oraciones
            oracion = oracion.strip() #esto es pa limpiar espacios 

            if not oracion:  
                continue

            palabras_oracion = len(oracion.split()) #se cuentan las palabras

            if palabras_actuales + palabras_oracion <= self.max_words: #se verifica que quepa en el chunk osea que sea de 250 palabras o menos
                chunk_actual.append(oracion)
                palabras_actuales += palabras_oracion

            else: #aqui es donde se corta el chunk
                texto_chunk = " ".join(chunk_actual)
                num_words = len(texto_chunk.split())
                num_tokens = len(
                    self.tokenizer.encode(
                        texto_chunk,
                        add_special_tokens=False
                    )
                )
                #se agrega el chunk a la lista de chunks
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-chunk-{posicion:03d}",
                    "fuente": metadata["source"],
                    "formato": metadata["format"],
                    "fenomeno": metadata.get("phenomenon"),
                    "posicion": posicion,
                    "num_words": num_words,
                    "num_tokens": num_tokens,
                    "texto": texto_chunk,
                    "idioma": metadata.get("language"),
                    "fecha": metadata.get("date"),
                    "titulo": metadata.get("title")
                })
                posicion += 1
                chunk_actual = [oracion]
                palabras_actuales = palabras_oracion
        #Esta parte de aca es por que el chunk queda abierto entonces toca ponerlo para que el ultimo chunk se agregue
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
                "fenomeno": metadata.get("phenomenon"),
                "posicion": posicion,
                "num_words": num_words,
                "num_tokens": num_tokens,
                "texto": texto_chunk,
                "idioma": metadata.get("language"),
                "fecha": metadata.get("date"),
                "titulo": metadata.get("title")

            })

        return chunks

class JSONExtractor:

    #Este es el metodo que pasa la data a texto plano y separa en metadata y texto
    def extract(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        # Construcción del texto plano, por ahora solo se toma body_text pero se puede hacer un
        #condicional para evaluar los campos que se que contienen el texto .
        body_text = data.get("body_text", "").strip()

        # Como metadata deje la información que se requiere en el formato del chunk.
        metadata = {
            "source": Path(file_path).name,
            "format": "json",
            "title": data.get("title"),
            "date": data.get("date"),
            # El JSON no lo trae explicito, para esto definamos una biblioteca o como lo hicieron?
            "language": data.get("language"),
            #Esto quedamos en que lo determinaba el path
            "phenomenon": data.get("phenomenon")
        }

        return {
            "content": body_text,
            "metadata": metadata

        }

if __name__ == "__main__":

    extractor = JSONExtractor() #no es nada raro, solo se crean las variables de las clases
    chunker = Chunker(
        max_words=250,
        language="en"
    )
    carpeta = Path("prueba")
    todos_los_chunks = []

    # Recorremos todos los JSON
    for indice_documento, archivo_json in enumerate(
            sorted(carpeta.glob("*.json")),
            start=1):

        # DOC-001, DOC-002, DOC-003...
        doc_id = f"DOC-{indice_documento:03d}"
        print(f"Procesando {archivo_json.name} -> {doc_id}") #Esto lo agregó la ia pero esta bonito jsjsjs
        documento = extractor.extract(archivo_json)
        chunks = chunker.split(documento, doc_id) #se toman los chunks del documento
        todos_los_chunks.extend(chunks) #Se agregan los chunks del documento a la lista de chunks que despues sera el json

    # Guardar todos los chunks
    with open("chunks.json", "w", encoding="utf-8") as file:

        json.dump(
            todos_los_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n--------------------------------------")
    print(f"Documentos procesados : {indice_documento}")
    print(f"Chunks generados      : {len(todos_los_chunks)}")
    print("Archivo generado      : chunks.json")