"""EncodeDB - Codificacion semantica e indexacion vectorial con FAISS.

CODEFEST AD ASTRA 2026 - Etapa 1 (Base de Conocimiento).

Restricciones del reto respetadas por este modulo:
  * Solo se emplean modelos encoder (familia BERT / XLM-RoBERTa). El uso de
    arquitecturas decoder esta prohibido en indexacion y recuperacion
    (Secciones 4.2 y 8.3 de la especificacion tecnica).
  * Similitud coseno mediante IndexFlatIP con vectores normalizados a norma
    unitaria (Secciones 5.2 y 8.2).
  * El orden de insercion en FAISS coincide exactamente con el orden de la
    lista de chunks devuelta, que es la que debe volcarse a metadata.jsonl
    (Seccion 1.4). Esta es la invariante critica del entregable.

Diseno: una instancia = un encoder = una subcarpeta encoder_<nombre>/.
Para usar varios encoders (Seccion 4.4), se instancia EncodeDB una vez por
modelo; cada instancia produce su propio indice independiente.

Dependencias: faiss-cpu (o faiss-gpu), sentence-transformers, numpy.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Campos exigidos por la Tabla 1 de la especificacion.
CAMPOS_OBLIGATORIOS: tuple[str, ...] = (
    "doc_id",
    "chunk_id",
    "fuente",
    "formato",
    "fenomeno",
    "posicion",
    "num_tokens",
    "texto",
)

# Prefijos de rol exigidos por algunas familias de encoders.
# E5 fue entrenado con "passage: " / "query: "; omitirlos degrada la
# recuperacion de forma silenciosa. BGE-M3 y LaBSE no usan prefijo.
_PREFIJOS_E5 = ("passage: ", "query: ")
_SIN_PREFIJO = ("", "")


class EncodeDB:
    """Codifica fragmentos y construye el indice FAISS de la base vectorial.

    Parametros
    ----------
    modelo : str
        Identificador HuggingFace del encoder. Debe ser una arquitectura
        encoder con licencia libre (Apache 2.0, MIT o CC BY).
    campo_texto : str
        Nombre del campo que contiene el texto del fragmento en los dicts
        de entrada. Por defecto "texto" (Tabla 1).
    prefijo_pasaje, prefijo_consulta : str | None
        Prefijos de rol. Si es None se detectan a partir del nombre del
        modelo; pasar "" fuerza la ausencia de prefijo.
    batch_size : int
        Tamano de lote de codificacion. 32-128 en GPU, 8-16 en CPU.
    device : str | None
        "cuda", "cpu", etc. None deja que la libreria elija.
    normalizar : bool
        Normalizacion a norma unitaria. Debe permanecer en True para que
        el producto interno equivalga a la similitud coseno.
    """

    def __init__(
        self,
        modelo: str = "intfloat/multilingual-e5-large",
        campo_texto: str = "texto",
        prefijo_pasaje: str | None = None,
        prefijo_consulta: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        normalizar: bool = True,
    ) -> None:
        self.nombre_modelo = modelo
        self.campo_texto = campo_texto
        self.batch_size = batch_size
        self.normalizar = normalizar

        if not normalizar:
            warnings.warn(
                "normalizar=False: IndexFlatIP calculara producto interno "
                "puro, no similitud coseno.",
                stacklevel=2,
            )

        self.encoder = SentenceTransformer(modelo, device=device)
        self.dim: int = self.encoder.get_sentence_embedding_dimension()
        self.max_tokens: int = int(getattr(self.encoder, "max_seq_length", 512))

        pas_def, con_def = self._detectar_prefijos(modelo)
        self.prefijo_pasaje = pas_def if prefijo_pasaje is None else prefijo_pasaje
        self.prefijo_consulta = con_def if prefijo_consulta is None else prefijo_consulta

        # Estado poblado por create_vector_db.
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Funciones de salida solicitadas
    # ------------------------------------------------------------------

    def create_vector_db(
        self,
        chunks: str | Path | Sequence[Any],
        ordenar_por: tuple[str, ...] | None = ("doc_id", "posicion"),
        mostrar_progreso: bool = True,
    ) -> tuple[faiss.Index, list[dict[str, Any]]]:
        """Codifica los fragmentos y construye el indice FAISS.

        Parametros
        ----------
        chunks : str | Path | Sequence
            Ruta a un archivo .jsonl con exactamente un objeto JSON por
            linea (cada objeto es un fragmento con los campos de la
            Tabla 1), o bien una secuencia ya cargada en memoria de dicts
            o cadenas de texto. Una cadena a este nivel se interpreta
            siempre como ruta de archivo, nunca como texto de fragmento.
        ordenar_por : tuple | None
            Campos por los que se ordena el corpus antes de indexar, para
            garantizar reproducibilidad entre ejecuciones. None conserva el
            orden de entrada. Si algun fragmento carece de esos campos, se
            conserva el orden de entrada y se emite un aviso.

        Retorna
        -------
        (index, chunks_ordenados)
            El indice FAISS listo y la lista de chunks en el MISMO orden en
            que fueron insertados. La posicion i de la lista corresponde al
            identificador interno i de FAISS, y es el orden exacto en que
            deben escribirse las lineas de metadata.jsonl.
        """
        registros = self._cargar_chunks(chunks)
        if not registros:
            raise ValueError("No se cargo ningun fragmento.")

        registros = self._ordenar(registros, ordenar_por)

        textos = [r[self.campo_texto] for r in registros]

        # num_tokens sobre el texto crudo, sin el prefijo de rol.
        conteos = self._contar_tokens(textos)
        excedidos = 0
        for reg, n in zip(registros, conteos):
            reg["num_tokens"] = n
            # Margen de 4 tokens por el prefijo de rol y los especiales.
            if n > self.max_tokens - 4:
                excedidos += 1
        if excedidos:
            warnings.warn(
                f"{excedidos} fragmento(s) superan el limite de "
                f"{self.max_tokens} tokens del encoder y seran truncados "
                "silenciosamente. Revise la estrategia de chunking.",
                stacklevel=2,
            )

        entradas = [self.prefijo_pasaje + t for t in textos]
        vectores = self.encoder.encode(
            entradas,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalizar,
            show_progress_bar=mostrar_progreso,
        )
        vectores = np.ascontiguousarray(vectores, dtype="float32")

        if vectores.shape[1] != self.dim:
            raise RuntimeError(
                f"Dimension inesperada: {vectores.shape[1]} != {self.dim}."
            )

        # Indice plano: busqueda exacta, sin entrenamiento ni parametros.
        # La insercion es secuencial: FAISS asigna los ids 0, 1, 2, ... en
        # ese orden. No paralelizar esta llamada.
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectores)

        if index.ntotal != len(registros):
            raise RuntimeError(
                f"Descuadre indice/metadata: {index.ntotal} vectores frente "
                f"a {len(registros)} fragmentos."
            )

        self.index = index
        self.chunks = registros
        return index, registros

    def encode_search(self, consulta: str | Sequence[str]) -> np.ndarray:
        """Codifica la consulta y devuelve su vector, listo para FAISS.

        Aplica el mismo encoder, el mismo prefijo de rol de consulta y la
        misma normalizacion que se usaron al indexar (Seccion 8.1).

        Retorna
        -------
        np.ndarray de forma (n, d) y dtype float32, donde n es 1 para una
        consulta simple. FAISS exige un arreglo bidimensional; se pueden
        pasar las 50 consultas en un solo lote, que ademas es la forma mas
        eficiente de consultar el indice.
        """
        if isinstance(consulta, str):
            consultas = [consulta]
        else:
            consultas = list(consulta)

        if not consultas or any(not isinstance(c, str) for c in consultas):
            raise ValueError("La consulta debe ser una cadena o una lista de cadenas.")

        entradas = [self.prefijo_consulta + c for c in consultas]
        vector = self.encoder.encode(
            entradas,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalizar,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vector, dtype="float32").reshape(-1, self.dim)

    # ------------------------------------------------------------------
    # Persistencia (entregable de la Seccion 1.4)
    # ------------------------------------------------------------------

    def guardar(self, directorio: str | Path) -> Path:
        """Escribe index.faiss y metadata.jsonl en el directorio indicado."""
        if self.index is None:
            raise RuntimeError("Primero debe ejecutarse create_vector_db().")

        destino = Path(directorio)
        destino.mkdir(parents=True, exist_ok=True)

        faltantes = sorted(
            {c for c in CAMPOS_OBLIGATORIOS for r in self.chunks if c not in r}
        )
        if faltantes:
            warnings.warn(
                "Campos obligatorios de la Tabla 1 ausentes en algun "
                f"fragmento: {faltantes}",
                stacklevel=2,
            )

        faiss.write_index(self.index, str(destino / "index.faiss"))
        with open(destino / "metadata.jsonl", "w", encoding="utf-8") as f:
            for registro in self.chunks:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return destino

    def cargar(self, directorio: str | Path) -> tuple[faiss.Index, list[dict[str, Any]]]:
        """Recarga un indice ya construido sin reindexar el corpus."""
        origen = Path(directorio)
        index = faiss.read_index(str(origen / "index.faiss"))
        with open(origen / "metadata.jsonl", encoding="utf-8") as f:
            chunks = [json.loads(linea) for linea in f if linea.strip()]

        if index.ntotal != len(chunks):
            raise RuntimeError(
                f"Descuadre indice/metadata: {index.ntotal} vectores frente "
                f"a {len(chunks)} lineas."
            )

        self.index = index
        self.chunks = chunks
        return index, chunks

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _detectar_prefijos(modelo: str) -> tuple[str, str]:
        return _PREFIJOS_E5 if "e5" in modelo.lower() else _SIN_PREFIJO

    def _cargar_chunks(
        self, origen: str | Path | Sequence[Any]
    ) -> list[dict[str, Any]]:
        """Despacha entre una ruta .jsonl y una secuencia en memoria."""
        if isinstance(origen, (str, Path)):
            return self._leer_jsonl(origen)
        return [self._normalizar_registro(c) for c in origen]

    def _leer_jsonl(self, ruta: str | Path) -> list[dict[str, Any]]:
        """Lee un archivo JSON Lines: un objeto JSON por linea.

        La lectura es secuencial y perezosa, de modo que el orden del
        archivo se conserva y no se carga el corpus completo en memoria
        antes de validarlo. Las lineas en blanco se ignoran.
        """
        ruta = Path(ruta)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo de chunks: {ruta}")

        registros: list[dict[str, Any]] = []
        with open(ruta, encoding="utf-8") as f:
            for n, linea in enumerate(f, start=1):
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    objeto = json.loads(linea)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON invalido en {ruta.name}, linea {n}: {e.msg}. "
                        "El formato esperado es un objeto JSON por linea."
                    ) from e
                if not isinstance(objeto, dict):
                    raise TypeError(
                        f"{ruta.name}, linea {n}: se esperaba un objeto JSON, "
                        f"se recibio {type(objeto).__name__}."
                    )
                try:
                    registros.append(self._normalizar_registro(objeto))
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"{ruta.name}, linea {n}: {e}") from e
        return registros

    def _normalizar_registro(self, chunk: Any) -> dict[str, Any]:
        """Acepta dicts o cadenas y devuelve siempre una copia como dict."""
        if isinstance(chunk, str):
            return {self.campo_texto: chunk}
        if not isinstance(chunk, dict):
            raise TypeError(f"Fragmento no soportado: {type(chunk).__name__}.")

        registro = dict(chunk)
        if self.campo_texto not in registro:
            for alias in ("texto", "text", "content", "contenido"):
                if alias in registro:
                    registro[self.campo_texto] = registro[alias]
                    break
            else:
                raise KeyError(
                    f"El fragmento no contiene el campo de texto "
                    f"'{self.campo_texto}'. Claves disponibles: "
                    f"{sorted(registro)}"
                )
        if not str(registro[self.campo_texto]).strip():
            raise ValueError("Se encontro un fragmento con texto vacio.")
        return registro

    @staticmethod
    def _ordenar(
        registros: list[dict[str, Any]], campos: tuple[str, ...] | None
    ) -> list[dict[str, Any]]:
        if not campos:
            return registros
        if not all(c in r for c in campos for r in registros):
            warnings.warn(
                f"No todos los fragmentos tienen los campos {campos}; se "
                "conserva el orden de entrada.",
                stacklevel=3,
            )
            return registros
        return sorted(registros, key=lambda r: tuple(r[c] for c in campos))

    def _contar_tokens(self, textos: list[str]) -> list[int]:
        tokenizer = self.encoder.tokenizer
        ids = tokenizer(textos, add_special_tokens=True)["input_ids"]
        return [len(x) for x in ids]

    def __repr__(self) -> str:
        total = self.index.ntotal if self.index is not None else 0
        return (
            f"EncodeDB(modelo='{self.nombre_modelo}', dim={self.dim}, "
            f"max_tokens={self.max_tokens}, vectores={total})"
        )
