"""
build_ngram_representations.py

Genera un unico CSV con rappresentazioni a n-grammi da un CSV con colonne:
- text
- label

Output:
- text
- char_2grams
- char_3grams
- char_4grams
- word_1grams
- word_2grams
- word_3grams
- word_4grams
- lemma_1grams
- lemma_2grams
- lemma_3grams
- lemma_4grams
- pos_1grams
- pos_2grams
- pos_3grams
- pos_4grams
- label

Ogni cella degli n-grammi contiene una lista JSON:
[
    {"ngram": "...", "freq": 3},
    {"ngram": "...", "freq": 1}
]

Il testo viene normalizzato.
Le stop word vengono eliminate.
Le frequenze NON vengono normalizzate.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter

import pandas as pd
import spacy


# ============================================================
# CLASSI DATI: Token, Sentence, Document
# ============================================================

class Token:
    def __init__(self, word, lemma, pos):
        self.word = word
        self.lemma = lemma
        self.pos = pos

    def num_chars(self):
        return len(self.word)

    def __repr__(self):
        return f"Token(word={self.word}, lemma={self.lemma}, pos={self.pos})"


class Sentence:
    def __init__(self, text, tokens=None):
        self.text = text
        if tokens is None:
            self.tokens = []
        else:
            self.tokens = tokens

    def add_token(self, token):
        self.tokens.append(token)

    def get_words(self):
        words = []
        for token in self.tokens:
            words.append(token.word)
        return words

    def get_lemmas(self):
        lemmas = []
        for token in self.tokens:
            lemmas.append(token.lemma)
        return lemmas

    def get_pos_tags(self):
        pos_tags = []
        for token in self.tokens:
            pos_tags.append(token.pos)
        return pos_tags

    def num_tokens(self):
        return len(self.tokens)

    def num_chars(self):
        if not self.tokens:
            return 0
        return sum(token.num_chars() for token in self.tokens) + (len(self.tokens) - 1)

    def __repr__(self):
        return f"Sentence(text={self.text}, tokens={self.tokens})"


class Document:
    def __init__(self, path, doc_id, split, label, sentences=None, features=None):
        self.path = path
        self.doc_id = doc_id
        self.split = split
        self.label = label

        if sentences is None:
            self.sentences = []
        else:
            self.sentences = sentences

        if features is None:
            self.features = {}
        else:
            self.features = features

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def get_tokens(self):
        tokens = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                tokens.append(token)
        return tokens

    def get_words(self):
        words = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                words.append(token.word)
        return words

    def get_lemmas(self):
        lemmas = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                lemmas.append(token.lemma)
        return lemmas

    def get_pos_tags(self):
        pos_tags = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                pos_tags.append(token.pos)
        return pos_tags

    def num_tokens(self):
        num_tokens = 0
        for sentence in self.sentences:
            num_tokens += sentence.num_tokens()
        return num_tokens

    def num_chars(self):
        num_chars = 0
        for sentence in self.sentences:
            num_chars += sentence.num_chars()
        return num_chars

    def num_sentences(self):
        return len(self.sentences)

    def __repr__(self):
        return (
            f"Document(path={self.path}, doc_id={self.doc_id}, "
            f"split={self.split}, label={self.label}, "
            f"num_sentences={len(self.sentences)})"
        )


# ============================================================
# NORMALIZZAZIONE TESTO
# ============================================================

def normalize_text(text):
    """
    Normalizza il testo prima dell'analisi:
    - Unicode NFC;
    - newline, tab e spazi multipli convertiti in spazio singolo;
    - spazi iniziali e finali rimossi.

    Non fa lowercase automatico.
    """

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# ============================================================
# COSTRUZIONE DOCUMENTI DA CSV text,label
# ============================================================

def build_document_from_text(text, label, doc_id, split, nlp):
    """
    Costruisce un Document partendo da un testo.

    Vengono eliminati:
    - spazi;
    - punteggiatura;
    - stop word.

    I token rimanenti vengono salvati come Token(word, lemma, pos).
    """

    document = Document(
        path=None,
        doc_id=str(doc_id),
        split=split,
        label=str(label)
    )

    spacy_doc = nlp(text)

    for spacy_sentence in spacy_doc.sents:
        sentence = Sentence(text=spacy_sentence.text)

        for spacy_token in spacy_sentence:
            if spacy_token.is_space:
                continue

            if spacy_token.is_punct:
                continue

            if spacy_token.is_stop:
                continue

            token = Token(
                word=spacy_token.text,
                lemma=spacy_token.lemma_,
                pos=spacy_token.pos_
            )

            sentence.add_token(token)

        if sentence.num_tokens() > 0:
            document.add_sentence(sentence)

    return document


def load_documents_from_csv(input_csv, split="train", spacy_model="it_core_news_sm"):
    """
    Legge un CSV con colonne text,label e restituisce una lista di Document.
    """

    dataframe = pd.read_csv(input_csv)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Il CSV deve contenere le colonne {required_columns}. "
            f"Colonne mancanti: {missing_columns}"
        )

    nlp = spacy.load(spacy_model)

    documents = []

    for doc_id, row in dataframe.iterrows():
        text = normalize_text(row["text"])
        label = row["label"]

        document = build_document_from_text(
            text=text,
            label=label,
            doc_id=doc_id,
            split=split,
            nlp=nlp
        )

        documents.append(document)

    return documents


# ============================================================
# ESTRAZIONE N-GRAMMI NON NORMALIZZATI
# ============================================================

def counter_to_json_list(counter):
    """
    Converte un Counter in una lista di dizionari ordinata.

    Ordinamento:
    - frequenza decrescente;
    - n-gramma crescente a parità di frequenza.
    """

    items = sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0])
    )

    return [
        {
            "ngram": ngram,
            "freq": int(freq)
        }
        for ngram, freq in items
    ]


def extract_char_ngrams_from_text(text, n):
    """
    Estrae n-grammi di caratteri da una stringa.
    Le frequenze sono conteggi assoluti.
    """

    counter = Counter()

    if len(text) < n:
        return []

    for i in range(0, len(text) - n + 1):
        ngram = text[i:i + n]
        counter[ngram] += 1

    return counter_to_json_list(counter)


def extract_sequence_ngrams(sequence, n):
    """
    Estrae n-grammi da una sequenza di:
    - parole;
    - lemmi;
    - POS.

    Le frequenze sono conteggi assoluti.
    """

    counter = Counter()

    if len(sequence) < n:
        return []

    for i in range(0, len(sequence) - n + 1):
        ngram_items = sequence[i:i + n]
        ngram = " ".join(ngram_items)
        counter[ngram] += 1

    return counter_to_json_list(counter)


# ============================================================
# COSTRUZIONE RIGA DATAFRAME
# ============================================================

def build_row_from_document(document):
    """
    Costruisce una riga del DataFrame finale.

    La colonna text contiene il testo effettivamente usato per gli n-grammi:
    normalizzato, senza punteggiatura e senza stop word.

    Le colonne degli n-grammi contengono stringhe JSON.
    """

    words = document.get_words()
    lemmas = document.get_lemmas()
    pos_tags = document.get_pos_tags()

    processed_text = " ".join(words)

    row = {}

    row["text"] = processed_text

    for n in range(2, 5):
        column_name = f"char_{n}grams"
        row[column_name] = json.dumps(
            extract_char_ngrams_from_text(processed_text, n),
            ensure_ascii=False
        )

    for n in range(1, 5):
        column_name = f"word_{n}grams"
        row[column_name] = json.dumps(
            extract_sequence_ngrams(words, n),
            ensure_ascii=False
        )

    for n in range(1, 5):
        column_name = f"lemma_{n}grams"
        row[column_name] = json.dumps(
            extract_sequence_ngrams(lemmas, n),
            ensure_ascii=False
        )

    for n in range(1, 5):
        column_name = f"pos_{n}grams"
        row[column_name] = json.dumps(
            extract_sequence_ngrams(pos_tags, n),
            ensure_ascii=False
        )

    row["label"] = document.label

    return row


def build_ngram_dataframe(documents):
    """
    Costruisce un unico DataFrame pandas.
    Ogni riga corrisponde a un documento.
    """

    rows = []

    for document in documents:
        row = build_row_from_document(document)
        rows.append(row)

    columns = [
        "text",
        "char_2grams",
        "char_3grams",
        "char_4grams",
        "word_1grams",
        "word_2grams",
        "word_3grams",
        "word_4grams",
        "lemma_1grams",
        "lemma_2grams",
        "lemma_3grams",
        "lemma_4grams",
        "pos_1grams",
        "pos_2grams",
        "pos_3grams",
        "pos_4grams",
        "label"
    ]

    dataframe = pd.DataFrame(rows, columns=columns)

    return dataframe


# ============================================================
# PIPELINE COMPLETA
# ============================================================

def create_ngram_csv(input_csv, output_csv, split="train", spacy_model="it_core_news_sm"):
    """
    Pipeline completa:
    CSV text,label
        -> normalizzazione testo
        -> analisi spaCy
        -> rimozione stop word
        -> costruzione Document/Sentence/Token
        -> estrazione n-grammi
        -> DataFrame pandas
        -> CSV finale
    """

    documents = load_documents_from_csv(
        input_csv=input_csv,
        split=split,
        spacy_model=spacy_model
    )

    dataframe = build_ngram_dataframe(documents)

    dataframe.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    print(f"File creato: {output_csv}")
    print(f"Documenti processati: {len(dataframe)}")
    print(f"Colonne generate: {len(dataframe.columns)}")

    return dataframe


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera un CSV con rappresentazioni a n-grammi da un CSV text,label."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Percorso del CSV di input con colonne text,label."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Percorso del CSV di output."
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Nome dello split: train, validation oppure test."
    )

    parser.add_argument(
        "--spacy-model",
        default="it_core_news_sm",
        help="Modello spaCy da usare. Default: it_core_news_sm."
    )

    args = parser.parse_args()

    create_ngram_csv(
        input_csv=args.input,
        output_csv=args.output,
        split=args.split,
        spacy_model=args.spacy_model
    )


if __name__ == "__main__":
    main()
