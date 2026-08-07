from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
import re

import pandas as pd
import tiktoken

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

    parts = re.split(
    r"\s*\|\s*|(?<=[.!?])\s+|\n+",
    text
    )
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

    print(
        "ANTES", count_tokens(body, encoder),
        "DESPUES", [count_tokens(x, encoder) for x in body_fragments]
    )

    units: list[str] = []
    for index, fragment in enumerate(body_fragments):
        tag = f"{label}: " if index == 0 else f"{label} (cont. {index + 1}): "
        units.append(f"{tag}{fragment}".strip())

    return units

def split_list_value(value: str) -> list[str]:
    if "|" not in value:
        return [value.strip()]

    return [item.strip() for item in value.split("|") if item.strip()]

def row_to_units(row: pd.Series, columns: list[str], encoder, max_tokens: int) -> list[str]:
    units: list[str] = []

    for column, value in zip(columns, row):
        value_str = str(value).strip()

        if not value_str or value_str.lower() == "nan":
            continue

        # Primero intentamos conservar la unidad completa si ya cabe.
        full_unit = f"{column}: {value_str}".strip()
        if count_tokens(full_unit, encoder) <= max_tokens:
            units.append(full_unit)
            continue

        # Si es una lista separada por '|', se procesa ítem por ítem.
        items = split_list_value(value_str)
        if len(items) > 1:
            for item_index, item in enumerate(items):
                label = column if item_index == 0 else f"{column} (cont.)"
                unit = f"{label}: {item}".strip()

                if count_tokens(unit, encoder) <= max_tokens:
                    units.append(unit)
                else:
                    units.extend(split_oversized_unit(label, item, encoder, max_tokens))
            continue

        # Si sigue siendo largo, lo cortamos por oraciones y luego por espacios.
        text_fragments = split_text_manually(value_str) or [value_str]

        for index, fragment in enumerate(text_fragments):
            label = column if index == 0 else f"{column} (cont.)"
            unit = f"{label}: {fragment}".strip()

            if count_tokens(unit, encoder) <= max_tokens:
                units.append(unit)
            else:
                units.extend(split_oversized_unit(label, fragment, encoder, max_tokens))
                
    return units

            

def tokens_of_units(units, encoder):
    return count_tokens(
        "\t".join(t for _, t in units),
        encoder
    )


def tokens_of_units(units: list[tuple[int, str]], encoder) -> int:
    return count_tokens(
        "\t".join(t for _, t in units),
        encoder
    )


def chunk_units(
    units: list[tuple[int, str]],
    encoder,
    config: ChunkConfig
) -> list[tuple[list[int], str]]:
    """
    Agrupa unidades en chunks respetando el máximo de tokens
    y el solapamiento por tokens.
    """

    chunks: list[tuple[list[int], str]] = []
    current_units: list[tuple[int, str]] = []

    SAFE_LIMIT = config.max_tokens - 20

    for row_id, unit in units:

        # ¿Cabe la unidad en el chunk actual?
        candidate_units = current_units + [(row_id, unit)]

        if current_units and tokens_of_units(candidate_units, encoder) > SAFE_LIMIT:

            # Guardar chunk actual
            chunks.append((
                sorted({r for r, _ in current_units}),
                "\t".join(t for _, t in current_units)
            ))

            # --------------------------
            # Construir overlap
            # --------------------------
            overlap_units: list[tuple[int, str]] = []
            overlap_tokens = 0

            for r, t in reversed(current_units):
                t_tokens = count_tokens(t, encoder)

                if overlap_units and overlap_tokens + t_tokens > config.overlap_tokens:
                    break

                overlap_units.insert(0, (r, t))
                overlap_tokens += t_tokens

                if overlap_tokens > config.overlap_tokens:
                    break

            # Intentar agregar la unidad nueva sobre el overlap
            candidate_units = overlap_units + [(row_id, unit)]

            if tokens_of_units(candidate_units, encoder) <= SAFE_LIMIT:
                current_units = candidate_units
            else:
                # Si ni con overlap cabe, descartar overlap
                current_units = [(row_id, unit)]

        else:
            current_units = candidate_units

    if current_units:
        chunks.append((
            sorted({r for r, _ in current_units}),
            "\t".join(t for _, t in current_units)
        ))

    return chunks

def _normalize_metadata_value(value, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value).strip()


def build_chunks(
    csv_path: str | Path,
    *,
    doc_id: str,
    fuente: str,
    fenomeno: int | str,
    fecha: str | None = None,
    titulo: str | None = None,
    formato: str = "csv",
    config: ChunkConfig | None = None,
) -> list[ChunkData]:
    """Construye los chunks a partir de un CSV."""

    config = config or ChunkConfig()
    csv_path = Path(csv_path)
    dataframe = pd.read_csv(csv_path)
    encoder = get_encoder(config.tokenizer_name)

    titulo_final = titulo or csv_path.name

    document_units: list[tuple[int, str]] = []

    for row_id, (_, row) in enumerate(dataframe.iterrows(), start=1):
        units = row_to_units(row, dataframe.columns.tolist(), encoder, config.max_tokens)
        document_units.extend((row_id, unit) for unit in units)

    document_chunks = chunk_units(document_units, encoder, config)

    records: list[ChunkData] = []
    for posicion, (_, texto) in enumerate(document_chunks, start=1):
        chunk_id = f"{doc_id}-chunk-{posicion:03d}"
        texto_final = texto.replace("\t", " | ")
        records.append(
            ChunkData(
                doc_id=doc_id,
                chunk_id=chunk_id,
                fuente=fuente,
                formato=formato,
                fenomeno=fenomeno,
                posicion=posicion,
                num_tokens=count_tokens(texto_final, encoder),
                texto=texto_final,
                idioma=detect_language(texto_final),
                fecha=_normalize_metadata_value(fecha, default=None),
                titulo=titulo_final,
            )
        )

    return records


def save_chunks_to_json(records: list[ChunkData], output_path: str | Path) -> Path:
    """Guarda la lista de chunks como un JSON con una lista de objetos."""

    output_path = Path(output_path)
    output_path.write_text(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    csv_path = Path("AMAZONUW_amazonunderworld-data.csv")
    records = build_chunks(
        csv_path,
        doc_id="DOC-120",
        fuente=csv_path.name,
        fenomeno=3,
        fecha="2024-12-31",
        titulo=csv_path.name,
    )
    output_path = save_chunks_to_json(records, Path("chunks.json"))

    #for record in records:
    #    print(record)

    print(f"Saved {len(records)} chunks to {output_path}")


if __name__ == "__main__":
    main()