import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSQuery:

    def __init__(self, index_path, metadata_path, model_name):
        """
        Carga el índice FAISS, los metadatos de los chunks
        y el encoder.
        """

        # Cargar índice FAISS
        self.index = faiss.read_index(index_path)

        # Cargar metadata JSONL
        self.metadata = self._cargar_metadata(metadata_path)

        # Cargar encoder
        self.encoder = SentenceTransformer(model_name)

    def _cargar_metadata(self, metadata_path):
        """
        Carga el archivo JSONL.

        La posición de cada elemento debe corresponder
        con la posición del vector en FAISS.
        """

        metadata = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            for linea in f:
                if linea.strip():
                    metadata.append(json.loads(linea))

        return metadata

    def generar_embedding(self, query):
        """
        Genera el embedding de la consulta utilizando
        el formato recomendado por multilingual-e5.
        """

        query_formateada = f"query: {query}"

        embedding = self.encoder.encode(
            query_formateada,
            normalize_embeddings=True
        )

        # FAISS espera una matriz de tamaño:
        # (numero_consultas, dimensiones)
        embedding = np.asarray(
            embedding,
            dtype="float32"
        ).reshape(1, -1)

        return embedding

    def buscar(self, query, k=10):
        """
        Busca los k chunks más cercanos en FAISS.
        """

        embedding = self.generar_embedding(query)

        distances, indices = self.index.search(
            embedding,
            k
        )

        resultados = []

        for rank, (distance, index) in enumerate(
            zip(distances[0], indices[0]),
            start=1
        ):

            # -1 significa que FAISS no encontró resultado
            if index == -1:
                continue

            chunk = self.metadata[index]

            resultados.append({
                "rank": rank,
                "distance": float(distance),
                "metadata": chunk
            })

        return resultados

    def construir_respuesta(
        self,
        query_id,
        resultados,
        numero_documentos=3
    ):
        """
        Construye la respuesta final.
    
        documents:
            Contiene los primeros N documentos diferentes
            encontrados entre los resultados de FAISS.
    
        fragments:
            Contiene todos los chunks recuperados por FAISS,
            manteniendo su ranking original.
        """
    
        documents = []
        fragments = []
    
        documentos_vistos = set()
    
        # -----------------------------------------
        # 1. Construir FRAGMENTS
        # -----------------------------------------
    
        for resultado in resultados:
    
            chunk = resultado["metadata"]
    
            fragments.append({
                "rank": resultado["rank"],
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["texto"]
            })
    
        # -----------------------------------------
        # 2. Construir DOCUMENTS
        # -----------------------------------------
    
        document_rank = 1
    
        for resultado in resultados:
    
            chunk = resultado["metadata"]
            doc_id = chunk["doc_id"]
    
            # Si ya encontramos este documento,
            # pasamos al siguiente chunk.
            if doc_id in documentos_vistos:
                continue
    
            # Guardamos el documento
            documentos_vistos.add(doc_id)
    
            documents.append({
                "rank": document_rank,
                "doc_id": doc_id
            })
    
            document_rank += 1
    
            # Ya tenemos los 3 documentos
            if len(documents) == numero_documentos:
                break
    
        # -----------------------------------------
        # 3. Construir respuesta
        # -----------------------------------------
    
        return {
            "query_id": query_id,
            "documents": documents,
            "fragments": fragments
        }

    def consultar(
        self,
        query_id,
        query,
        k=10,
        numero_documentos=3
    ):
        """
        Ejecuta una consulta completa.
        """

        resultados = self.buscar(
            query,
            k=k
        )

        respuesta = self.construir_respuesta(
            query_id=query_id,
            resultados=resultados,
            numero_documentos=numero_documentos
        )

        return respuesta

RUTA_INDEX = '/entrega/base_vectorial/encoder_multilingual/index.faiss'
RUTA_METADATA = '/entrega/base_vectorial/encoder_multilingual/metadata.jsonl'

buscador = FAISSQuery(
        index_path=RUTA_INDEX,
        metadata_path=RUTA_METADATA,
        model_name="intfloat/multilingual-e5-base"
    )

query = "add astra 2024"

respuesta = buscador.consultar(
        query_id="q001",
        query=query,
        k=10,
        numero_documentos=3
    )

print(json.dumps(
        respuesta,
        ensure_ascii=False,
        indent=2
    ))