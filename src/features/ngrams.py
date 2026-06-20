import argparse
import json
import re
import unicodedata
from collections import Counter

import pandas as pd
import spacy


CHAR_NS = range(2, 5)
SEQ_NS = range(1, 5)
SEQ_TYPES = ("word", "lemma", "pos")

OUTPUT_COLUMNS = [
    "text",
    *(f"char_{n}grams" for n in CHAR_NS),
    *(f"{kind}_{n}grams" for kind in SEQ_TYPES for n in SEQ_NS),
    "label",
]


def normalize_text(text):
    """Normalizza Unicode e spazi, senza trasformare il testo in lowercase."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def valid_tokens(doc):
    """Tiene solo token utili: niente spazi, punteggiatura o stop word."""
    return [t for t in doc if not (t.is_space or t.is_punct or t.is_stop)]


def ngram_counter(items, n, joiner=" "):
    """Conta gli n-grammi di una sequenza usando Counter della libreria standard."""
    return Counter(joiner.join(items[i : i + n]) for i in range(len(items) - n + 1))


def char_ngram_counter(text, n):
    """Conta gli n-grammi di caratteri, includendo anche gli spazi tra parole."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def counter_to_json(counter, relative=True):
    """Converte un Counter in JSON ordinato: frequenza decrescente, poi n-gramma."""
    total = sum(counter.values())
    data = [
        {"ngram": ngram, "freq": freq / total if relative and total else int(freq)}
        for ngram, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    ]
    return json.dumps(data, ensure_ascii=False)


def build_row(doc, label):
    """Estrae testo processato, n-grammi di caratteri, parole, lemmi e POS."""
    tokens = valid_tokens(doc)
    values = {
        "word": [t.text for t in tokens],
        "lemma": [t.lemma_ for t in tokens],
        "pos": [t.pos_ for t in tokens],
    }
    text = " ".join(values["word"])

    row = {"text": text, "label": str(label)}
    row.update({f"char_{n}grams": counter_to_json(char_ngram_counter(text, n)) for n in CHAR_NS})
    row.update({
        f"{kind}_{n}grams": counter_to_json(ngram_counter(values[kind], n))
        for kind in SEQ_TYPES
        for n in SEQ_NS
    })
    return row


def create_ngram_csv(input_csv, output_csv, split="train", spacy_model="it_core_news_sm"):
    """
    Pipeline completa:
    CSV text,label -> normalizzazione -> spaCy -> filtro token -> n-grammi -> CSV.

    Il parametro split è mantenuto per compatibilità con il vecchio script,
    ma non modifica l'output perché non era scritto nel CSV finale.
    """
    df = pd.read_csv(input_csv)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    texts = df["text"].map(normalize_text).tolist()
    nlp = spacy.load(spacy_model)

    rows = [build_row(doc, label) for doc, label in zip(nlp.pipe(texts), df["label"])]
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"File creato: {output_csv}")
    print(f"Documenti processati: {len(result)}")
    print(f"Colonne generate: {len(result.columns)}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Genera un CSV con rappresentazioni a n-grammi da un CSV text,label."
    )
    parser.add_argument("--input", required=True, help="CSV di input con colonne text,label.")
    parser.add_argument("--output", required=True, help="CSV di output.")
    parser.add_argument("--split", default="train", help="Mantenuto per compatibilità.")
    parser.add_argument("--spacy-model", default="it_core_news_sm", help="Modello spaCy da usare.")
    args = parser.parse_args()

    create_ngram_csv(args.input, args.output, args.split, args.spacy_model)


if __name__ == "__main__":
    main()
