from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
import tiktoken


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
    """Representa la informacion de cada chunk.

    `text` contiene el texto plano y el resto de campos son metadatos.
    """

    doc_id: int
    chunk_id: int
    text: str
    source_row_ids: list[int]

@dataclass
class ChunkConfig:
    """Configuracion basica del chunking.

    Define los tokens maximos por chunk, la cantidad de tokens de solapamiento
    entre chunks y el nombre del tokenizer a usar.
    """

    max_tokens: int = 256
    overlap_tokens: int = 75
    tokenizer_name: str = "cl100k_base"

def get_encoder(tokenizer_name: str):
    """Devuelve el encoder de tiktoken para el tokenizer indicado."""

    try:
        return tiktoken.get_encoding(tokenizer_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    """Cuenta la cantidad de tokens que contiene un texto."""

    return len(encoder.encode(text))

def split_text_manually(text: str) -> list[str]:
    """Divide el texto en partes mas pequenas usando puntuacion y saltos de linea."""

    text = str(text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned_parts = [part.strip() for part in parts if part.strip()]
    return cleaned_parts or [text]

def split_oversized_body(body: str, encoder, max_tokens: int) -> list[str]:
    """Divide el cuerpo de un valor (sin prefijo) en fragmentos que quepan en max_tokens.

    Intenta primero cortar por '|' (listas), y si un item individual sigue
    siendo demasiado largo, cae a corte por espacios.
    """

    if count_tokens(body, encoder) <= max_tokens:
        return [body]

    # Preferir cortar por "|" si el valor es una lista de items
    if "|" in body:
        items = body.split("|")
    else:
        items = body.split()  # fallback: palabras sueltas

    separator = "|" if "|" in body else " "
    fragments: list[str] = []
    current_items: list[str] = []

    for item in items:
        candidate = current_items + [item]
        candidate_text = separator.join(candidate)

        if current_items and count_tokens(candidate_text, encoder) > max_tokens:
            fragments.append(separator.join(current_items))
            current_items = [item]
        else:
            current_items = candidate

    if current_items:
        fragments.append(separator.join(current_items))

    # Si un item individual (separado por "|") sigue siendo enorme, partirlo por espacios
    final_fragments: list[str] = []
    for fragment in fragments:
        if count_tokens(fragment, encoder) > max_tokens and " " in fragment:
            words = fragment.split()
            current_words: list[str] = []
            for word in words:
                candidate_words = current_words + [word]
                if current_words and count_tokens(" ".join(candidate_words), encoder) > max_tokens:
                    final_fragments.append(" ".join(current_words))
                    current_words = [word]
                else:
                    current_words = candidate_words
            if current_words:
                final_fragments.append(" ".join(current_words))
        else:
            final_fragments.append(fragment)

    return final_fragments or [body]


def split_oversized_unit(label: str, body: str, encoder, max_tokens: int) -> list[str]:
    """Divide una unidad 'columna: valor' demasiado larga, conservando el label en cada fragmento."""

    body_fragments = split_oversized_body(body, encoder, max_tokens)

    units: list[str] = []
    for index, fragment in enumerate(body_fragments):
        tag = f"{label}: " if index == 0 else f"{label} (cont. {index + 1}): "
        units.append(f"{tag}{fragment}".strip())

    return units

def row_to_units(row: pd.Series, columns: list[str], encoder, max_tokens: int) -> list[str]:
    units: list[str] = []

    for column, value in zip(columns, row):
        value_str = str(value).strip()
        full_unit = f"{column}: {value_str}".strip()

        # Si el valor completo ya entra, no lo toques -> evita cortes falsos por abreviaturas
        if count_tokens(full_unit, encoder) <= max_tokens:
            units.append(full_unit)
            continue

        # Solo si excede el límite, ahí sí lo partimos en oraciones/líneas
        text_fragments = split_text_manually(value_str)
        if not text_fragments:
            text_fragments = [""]

        for index, fragment in enumerate(text_fragments):
            label = column if index == 0 else f"{column} (cont.)"
            unit = f"{label}: {fragment}".strip()

            if count_tokens(unit, encoder) > max_tokens:
                units.extend(split_oversized_unit(label, fragment, encoder, max_tokens))
            else:
                units.append(unit)

    return units

def chunk_units(units: list[tuple[int, str]], encoder, config: ChunkConfig) -> list[tuple[list[int], str]]:
    """Agrupa unidades en chunks respetando el maximo de tokens y el solapamiento por tokens.

    El overlap toma unidades desde el final del chunk actual hacia atras hasta
    sumar aproximadamente `overlap_tokens`. Si la ultima unidad sola ya supera
    ese presupuesto, se incluye completa de todos modos (nunca se trunca una
    unidad, para no volver a romper el label de columna).
    """

    chunks: list[tuple[list[int], str]] = []
    current_units: list[tuple[int, str]] = []
    current_tokens = 0

    for row_id, unit in units:
        unit_tokens = count_tokens(unit, encoder)

        if current_units and current_tokens + unit_tokens > config.max_tokens:
            chunks.append((sorted({r for r, _ in current_units}), "\t".join(t for _, t in current_units)))

            # Armar el overlap por presupuesto de tokens, tomando desde el final
            overlap_units: list[tuple[int, str]] = []
            overlap_tokens_sum = 0

            for r, t in reversed(current_units):
                t_tokens = count_tokens(t, encoder)

                # Si ya hay overlap acumulado y agregar esta unidad se pasa del presupuesto, paramos
                if overlap_units and overlap_tokens_sum + t_tokens > config.overlap_tokens:
                    break

                overlap_units.insert(0, (r, t))
                overlap_tokens_sum += t_tokens

                # Si esta primera unidad sola ya excede el presupuesto, la dejamos
                # (mejor pasarse que perder el contexto) y no seguimos agregando mas
                if overlap_tokens_sum > config.overlap_tokens:
                    break

            current_units = overlap_units
            current_tokens = overlap_tokens_sum
        current_units.append((row_id, unit))
        current_tokens += unit_tokens

    if current_units:
        chunks.append((sorted({r for r, _ in current_units}), "\t".join(t for _, t in current_units)))

    return chunks

def build_chunks(csv_path: str | Path, config: ChunkConfig | None = None) -> list[ChunkData]:
    """Construye los chunks a partir de un CSV."""

    config = config or ChunkConfig()
    dataframe = pd.read_csv(csv_path)
    encoder = get_encoder(config.tokenizer_name)

    doc_id = 1
    document_units: list[tuple[int, str]] = []

    for row_id, (_, row) in enumerate(dataframe.iterrows(), start=1):
        units = row_to_units(row, dataframe.columns.tolist(), encoder, config.max_tokens)
        document_units.extend((row_id, unit) for unit in units)

    document_chunks = chunk_units(document_units, encoder, config)

    records: list[ChunkData] = []
    for chunk_id, (source_row_ids, text) in enumerate(document_chunks, start=1):
        records.append(ChunkData(doc_id=doc_id, chunk_id=chunk_id, text=text, source_row_ids=source_row_ids))

    return records


def main() -> None:
    csv_path = Path("AIINDEX_clinicaltrials-robotics-csv.csv")
    records = build_chunks(csv_path)

    for record in records:
        print(record)


if __name__ == "__main__":
    main()