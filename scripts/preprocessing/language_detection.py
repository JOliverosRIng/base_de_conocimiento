from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable, Tuple

# `pandas` is only needed when scanning CSVs; import it lazily to allow
# importing this module in environments without pandas installed.

# Prefer a lightweight third-party detector when available
try:
    from langdetect import detect  # type: ignore

    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

_SPANISH_HINTS = {
    "el",
    "la",
    "de",
    "del",
    "y",
    "que",
    "para",
    "en",
    "por",
    "con",
    "una",
    "un",
    "los",
    "las",
    "pais",
    "region",
    "indicador",
    "desplazamiento",
    "forzado",
}

_ENGLISH_HINTS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "with",
    "study",
    "trial",
    "data",
}


def detect_language(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "unknown"

    # If langdetect is available, use it (more robust than simple hints)
    if _HAS_LANGDETECT:
        try:
            lang = detect(text)
            if lang in ("es", "en"):
                return lang
            # Return detected code for other languages as well
            return lang
        except Exception:
            # Fall back to simple heuristic on any error
            pass

    normalized = text.lower()
    spanish_score = sum(1 for hint in _SPANISH_HINTS if hint in normalized)
    english_score = sum(1 for hint in _ENGLISH_HINTS if hint in normalized)

    if spanish_score > english_score:
        return "es"
    if english_score > spanish_score:
        return "en"
    return "unknown"


def detect_languages(texts: Iterable[str]) -> Counter:
    counts = Counter()
    for t in texts:
        t = str(t).strip()
        if not t:
            continue
        lang = detect_language(t)
        counts[lang] += 1
    return counts


def detect_csv_language(path: str, column: str | None = None, sample_n: int = 200) -> Tuple[str, Counter]:
    import pandas as pd

    df = pd.read_csv(path)

    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")
        texts = df[column].dropna().astype(str).head(sample_n).tolist()
    else:
        # Concatenate all columns per row to get a representative text
        texts = df.dropna(how="all").astype(str).agg(" ".join, axis=1).head(sample_n).tolist()

    counts = detect_languages(texts)
    most_common = counts.most_common(1)
    top_lang = most_common[0][0] if most_common else "unknown"
    return top_lang, counts


def _cli():
    parser = argparse.ArgumentParser(description="Quick CSV language scanner (lightweight)")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--column", "-c", help="Column to sample for language detection")
    parser.add_argument("--sample", "-n", type=int, default=200, help="Number of rows to sample")
    args = parser.parse_args()

    top_lang, counts = detect_csv_language(args.csv, column=args.column, sample_n=args.sample)
    print(f"Top language: {top_lang}")
    print("Counts:")
    for lang, cnt in counts.most_common():
        print(f"  {lang}: {cnt}")


if __name__ == "__main__":
    _cli()
