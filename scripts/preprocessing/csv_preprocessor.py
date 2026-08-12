from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from transformers import AutoTokenizer
from language_detection import detect_language


'''
Diferencia: para los chunks tengo en cuenta la cantidad de tokens, no de palabrtas,
no obstante me aseguro de que cada unidad no supere 
la cantidad de tokens, y si lo hace, 
la divido en fragmentos mas pequenos por espacios. 
Ademas, agrego un solapamiento entre chunks 
para asegurar que no se pierda informacion importante.

'''
@dataclass
class ChunkData:
    """Representa la informacion final de cada chunk."""

    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int | str
    posicion: int
    num_tokens: int
    texto: str
    idioma: str
    fecha: str | None
    titulo: str

@dataclass
class ChunkConfig:
    """Configuracion basica del chunking.

    Define los tokens maximos por chunk, la cantidad de tokens de solapamiento
    entre chunks y el nombre del tokenizer a usar.
    """

    max_tokens: int = 256
    overlap_tokens: int = 75
    tokenizer_name: str = "intfloat/multilingual-e5-base"


class CSVProcessor:
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()
        self.encoder = self.get_encoder(self.config.tokenizer_name)

    @staticmethod
    def get_encoder(tokenizer_name: str):
        """Devuelve el tokenizer de Hugging Face para el modelo indicado."""

        return AutoTokenizer.from_pretrained(tokenizer_name)

    @staticmethod
    def count_tokens(text: str, encoder) -> int:
        """Cuenta la cantidad de tokens que contiene un texto."""

        return len(encoder.encode(text, add_special_tokens=False))

    @staticmethod
    def split_text_manually(text: str) -> list[str]:
        """Divide el texto en partes mas pequenas usando puntuacion y saltos de linea."""

        text = str(text).strip()
        if not text:
            return []

        parts = re.split(r"\s*\|\s*|(?<=[.!?])\s+|\n+", text)
        cleaned_parts = [part.strip() for part in parts if part.strip()]
        return cleaned_parts or [text]

    def split_oversized_body(self, body: str, max_tokens: int) -> list[str]:
        """Divide el cuerpo de un valor (sin prefijo) en fragmentos que quepan en max_tokens."""

        if self.count_tokens(body, self.encoder) <= max_tokens:
            return [body]

        if "|" in body:
            items = body.split("|")
        else:
            items = body.split()

        separator = "|" if "|" in body else " "
        fragments: list[str] = []
        current_items: list[str] = []

        for item in items:
            candidate = current_items + [item]
            candidate_text = separator.join(candidate)

            if current_items and self.count_tokens(candidate_text, self.encoder) > max_tokens:
                fragments.append(separator.join(current_items))
                current_items = [item]
            else:
                current_items = candidate

        if current_items:
            fragments.append(separator.join(current_items))

        final_fragments: list[str] = []
        for fragment in fragments:
            if self.count_tokens(fragment, self.encoder) > max_tokens and " " in fragment:
                words = fragment.split()
                current_words: list[str] = []
                for word in words:
                    candidate_words = current_words + [word]
                    if current_words and self.count_tokens(" ".join(candidate_words), self.encoder) > max_tokens:
                        final_fragments.append(" ".join(current_words))
                        current_words = [word]
                    else:
                        current_words = candidate_words
                if current_words:
                    final_fragments.append(" ".join(current_words))
            elif self.count_tokens(fragment, self.encoder) > max_tokens:
                final_fragments.extend(self._split_by_characters(fragment, max_tokens))
            else:
                final_fragments.append(fragment)

        return final_fragments or [body]

    def _split_by_characters(self, text: str, max_tokens: int) -> list[str]:
        """Último recurso: parte por longitud de texto cuando no hay espacios útiles."""

        if self.count_tokens(text, self.encoder) <= max_tokens:
            return [text]

        pieces: list[str] = []
        current_piece = ""

        for character in text:
            candidate = current_piece + character
            if current_piece and self.count_tokens(candidate, self.encoder) > max_tokens:
                pieces.append(current_piece)
                current_piece = character
            else:
                current_piece = candidate

        if current_piece:
            pieces.append(current_piece)

        return pieces or [text]

    def split_oversized_unit(self, label: str, body: str, max_tokens: int) -> list[str]:
        # estimación conservadora del tag más largo posible
        tag_estimate = f"{label} (cont. 99): "
        tag_tokens = self.count_tokens(tag_estimate, self.encoder)
        body_budget = max(max_tokens - tag_tokens, 1)

        body_fragments = self.split_oversized_body(body, body_budget)

        units: list[str] = []
        for index, fragment in enumerate(body_fragments):
            tag = f"{label}: " if index == 0 else f"{label} (cont. {index + 1}): "
            unit = f"{tag}{fragment}".strip()
            if self.count_tokens(unit, self.encoder) > max_tokens:  # red de seguridad
                budget = max(max_tokens - self.count_tokens(tag, self.encoder), 1)
                fragment = self._split_by_characters(fragment, budget)[0]
                unit = f"{tag}{fragment}".strip()
            units.append(unit)
        return units

    @staticmethod
    def split_list_value(value: str) -> list[str]:
        if "|" not in value:
            return [value.strip()]

        return [item.strip() for item in value.split("|") if item.strip()]

    def row_to_units(self, row: pd.Series, columns: list[str], max_tokens: int) -> list[str]:
        units: list[str] = []

        for column, value in zip(columns, row):
            value_str = str(value).strip()

            if not value_str or value_str.lower() == "nan":
                continue

            full_unit = f"{column}: {value_str}".strip()
            if self.count_tokens(full_unit, self.encoder) <= max_tokens:
                units.append(full_unit)
                continue

            items = self.split_list_value(value_str)
            if len(items) > 1:
                for item_index, item in enumerate(items):
                    label = column if item_index == 0 else f"{column} (cont. {item_index + 1})"
                    unit = f"{label}: {item}".strip()

                    if self.count_tokens(unit, self.encoder) <= max_tokens:
                        units.append(unit)
                    else:
                        units.extend(self.split_oversized_unit(label, item, max_tokens))
                continue

            text_fragments = self.split_text_manually(value_str) or [value_str]

            for index, fragment in enumerate(text_fragments):
                label = column if index == 0 else f"{column} (cont. {index + 1})"
                unit = f"{label}: {fragment}".strip()

                if self.count_tokens(unit, self.encoder) <= max_tokens:
                    units.append(unit)
                else:
                    units.extend(self.split_oversized_unit(label, fragment, max_tokens))

        return units

    def tokens_of_units(self, units: list[tuple[int, str]]) -> int:
        return self.count_tokens(" | ".join(text for _, text in units), self.encoder)
    
    def chunk_units(
        self,
        units: list[tuple[int, str]],
        config: ChunkConfig | None = None,
    ) -> list[tuple[list[int], str]]:
        """Agrupa unidades en chunks respetando el máximo de tokens y el solapamiento."""

        config = config or self.config
        chunks: list[tuple[list[int], str]] = []
        current_units: list[tuple[int, str]] = []
        safe_limit = config.max_tokens

        for row_id, unit in units:
            candidate_units = current_units + [(row_id, unit)]

            if current_units and self.tokens_of_units(candidate_units) > safe_limit:
                chunks.append((
                    sorted({r for r, _ in current_units}),
                    " | ".join(text for _, text in current_units),
                ))

                overlap_units: list[tuple[int, str]] = []
                overlap_tokens = 0

                for row_index, text in reversed(current_units):
                    text_tokens = self.count_tokens(text, self.encoder)

                    if overlap_units and overlap_tokens + text_tokens > config.overlap_tokens:
                        break

                    overlap_units.insert(0, (row_index, text))
                    overlap_tokens += text_tokens

                    if overlap_tokens > config.overlap_tokens:
                        break

                candidate_units = overlap_units + [(row_id, unit)]

                if self.tokens_of_units(candidate_units) <= safe_limit:
                    current_units = candidate_units
                else:
                    current_units = [(row_id, unit)]
            else:
                current_units = candidate_units

        if current_units:
            chunks.append((
                sorted({r for r, _ in current_units}),
                " | ".join(text for _, text in current_units),
            ))

        return chunks

    @staticmethod
    def _normalize_metadata_value(value, default: str | None = "") -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        text = str(value).strip()
        return text if text else default

    @staticmethod
    def _asignar_doc_id(fuente: str) -> str:
        """Genera un doc_id estable a partir de la fuente."""

        hash_corto = hashlib.sha1(fuente.encode("utf-8")).hexdigest()[:8]
        return f"DOC-{hash_corto}"

    @staticmethod
    def _load_dataframe(input_path: str | Path) -> pd.DataFrame:
        path = Path(input_path)
        suffix = path.suffix.lower()

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        return pd.read_csv(path)

    @staticmethod
    def extraer_metadatos_archivo(csv_path: str | Path) -> dict[str, str | None]:
        """Extrae metadatos base del archivo a partir de la ruta y la fecha de modificacion."""

        path = Path(csv_path)
        stat_info = path.stat()
        fuente = path.name
        titulo = path.stem.replace("_", " ").strip() or path.name
        fecha = datetime.fromtimestamp(stat_info.st_mtime).date().isoformat()
        formato = path.suffix.lstrip(".").lower() or "csv"
        doc_id = CSVProcessor._asignar_doc_id(fuente)
        return {
            "doc_id": doc_id,
            "fuente": fuente,
            "formato": formato,
            "titulo": titulo,
            "fecha": fecha,
        }

    def build_chunks(
        self,
        csv_path: str | Path,
        fenomeno: int | str,
        config: ChunkConfig | None = None,
    ) -> list[ChunkData]:
        """Construye los chunks a partir de un CSV o XLSX."""

        config = config or self.config
        csv_path = Path(csv_path)
        dataframe = self._load_dataframe(csv_path)

        metadata_archivo = self.extraer_metadatos_archivo(csv_path)
        doc_id = metadata_archivo["doc_id"]
        fuente = metadata_archivo["fuente"] or csv_path.name
        formato = metadata_archivo["formato"] or csv_path.suffix.lstrip(".").lower() or "csv"
        titulo_final = metadata_archivo["titulo"] or csv_path.name
        fecha_final = metadata_archivo["fecha"]

        document_units: list[tuple[int, str]] = []

        for row_id, (_, row) in enumerate(dataframe.iterrows(), start=1):
            units = self.row_to_units(row, dataframe.columns.tolist(), config.max_tokens)
            document_units.extend((row_id, unit) for unit in units)

        document_chunks = self.chunk_units(document_units, config)

        records: list[ChunkData] = []
        for posicion, (_, texto) in enumerate(document_chunks, start=1):
            chunk_id = f"{doc_id}-chunk-{posicion:03d}"
            texto_final = texto
            records.append(
                ChunkData(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    fuente=fuente,
                    formato=formato,
                    fenomeno=fenomeno,
                    posicion=posicion,
                    num_tokens=self.count_tokens(texto_final, self.encoder),
                    texto=texto_final,
                    idioma=detect_language(texto_final),
                    fecha=self._normalize_metadata_value(fecha_final, default=None),
                    titulo=titulo_final,
                )
            )

        return records

    @staticmethod
    def _serialize_records(records: list[ChunkData] | list[dict[str, object]]) -> list[dict[str, object]]:
        """Convierte registros de chunk a una lista serializable en JSON."""

        serializable_records: list[dict[str, object]] = []
        for record in records:
            if isinstance(record, ChunkData):
                serializable_records.append(asdict(record))
            elif isinstance(record, dict):
                serializable_records.append(record)
            else:
                serializable_records.append(dict(record))
        return serializable_records

    @staticmethod
    def save_chunks_to_json(
        records: list[ChunkData] | list[dict[str, object]],
        output_path: str | Path,
    ) -> Path:
        """Guarda la lista de chunks como un JSON con una lista de objetos."""

        output_path = Path(output_path)
        payload = CSVProcessor._serialize_records(records)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def verify_coverage(
        self,
        csv_path: str | Path,
        records: list[ChunkData],
        config: ChunkConfig | None = None,
    ) -> dict:
        """Verifica que toda la información del archivo original esté presente
        en los chunks finales generados (records).

        Reconstruye las unidades esperadas (fila -> fragmentos "columna: valor")
        y chequea que cada una aparezca en el texto de al menos un chunk.
        """
        config = config or self.config
        dataframe = self._load_dataframe(Path(csv_path))

        # 1. Reconstruir las unidades esperadas, igual que en build_chunks
        unidades_esperadas: list[tuple[int, str]] = []
        for row_id, (_, row) in enumerate(dataframe.iterrows(), start=1):
            units = self.row_to_units(row, dataframe.columns.tolist(), config.max_tokens)
            unidades_esperadas.extend((row_id, unit) for unit in units)

        # 2. Texto combinado de todos los chunks realmente guardados
        texto_combinado = " ".join(record.texto for record in records)

        # 3. Buscar unidades que no aparezcan en ningún chunk
        unidades_faltantes: list[tuple[int, str]] = []
        filas_con_datos_faltantes: set[int] = set()

        for row_id, unit in unidades_esperadas:
            if unit not in texto_combinado:
                unidades_faltantes.append((row_id, unit))
                filas_con_datos_faltantes.add(row_id)

        total_filas = len(dataframe)
        total_unidades = len(unidades_esperadas)
        filas_ok = total_filas - len(filas_con_datos_faltantes)

        reporte = {
            "total_filas_dataframe": total_filas,
            "total_unidades_esperadas": total_unidades,
            "total_chunks_generados": len(records),
            "filas_completas": filas_ok,
            "filas_con_datos_faltantes": sorted(filas_con_datos_faltantes),
            "unidades_faltantes": unidades_faltantes,
            "cobertura_filas_pct": round(filas_ok / total_filas * 100, 2) if total_filas else 0,
            "cobertura_unidades_pct": round(
                (total_unidades - len(unidades_faltantes)) / total_unidades * 100, 2
            ) if total_unidades else 0,
            "ok": not unidades_faltantes,
        }
        return reporte

    @staticmethod
    def print_coverage_report(reporte: dict) -> None:
        """Imprime el reporte de cobertura de forma legible."""
        print("=== Reporte de cobertura ===")
        print(f"Filas en el archivo:        {reporte['total_filas_dataframe']}")
        print(f"Unidades esperadas:         {reporte['total_unidades_esperadas']}")
        print(f"Chunks generados:           {reporte['total_chunks_generados']}")
        print(f"Filas completas:            {reporte['filas_completas']}")
        print(f"Cobertura por filas:        {reporte['cobertura_filas_pct']}%")
        print(f"Cobertura por unidades:     {reporte['cobertura_unidades_pct']}%")

        if reporte["ok"]:
            print(" Toda la información fue capturada correctamente.")
        else:
            print(f"Faltan {len(reporte['unidades_faltantes'])} unidades "
                f"en {len(reporte['filas_con_datos_faltantes'])} filas.")
            for row_id, unit in reporte["unidades_faltantes"][:20]:
                print(f"  - Fila {row_id}: {unit[:120]}")
            if len(reporte["unidades_faltantes"]) > 20:
                print(f"  ... y {len(reporte['unidades_faltantes']) - 20} más.")
def build_chunks(
    csv_path: str | Path,
    fenomeno: int | str,
    config: ChunkConfig | None = None,
) -> list[ChunkData]:
    """Compatibilidad hacia atras para el flujo de chunks del CSV."""

    return CSVProcessor(config).build_chunks(csv_path, fenomeno, config=config)


def save_chunks_to_json(records: list[ChunkData], output_path: str | Path) -> Path:
    """Compatibilidad hacia atras para guardar chunks en JSON."""

    return CSVProcessor().save_chunks_to_json(records, output_path)


def process_csv_file(
    csv_path: str | Path,
    fenomeno: int | str,
) -> list[dict[str, object]]:
    """Procesa un CSV o XLSX y devuelve la lista de chunks como objetos JSON."""

    processor = CSVProcessor()
    records = processor.build_chunks(csv_path, fenomeno=fenomeno)
    return [asdict(record) for record in records]


def main() -> None:
    r = process_csv_file("AIINDEX_clinicaltrials-robotics-csv.csv", fenomeno=1)
    save_chunks_to_json(r, "AIINDEX_clinicaltrials-robotics-chunks.json")
    verify = CSVProcessor().verify_coverage("AIINDEX_clinicaltrials-robotics-csv.csv", [ChunkData(**rec) for rec in r])
    CSVProcessor.print_coverage_report(verify)

if __name__ == "__main__":
    main()