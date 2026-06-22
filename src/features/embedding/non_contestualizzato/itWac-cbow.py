from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess


# ===================== CONFIGURAZIONE =====================

SQLITE_PATH = "data/features/embedding/non_contestualizzato/itWaC-cbow/modello/itwac128.sqlite"
MODEL_PATH = "data/features/embedding/non_contestualizzato/itWaC-cbow/modello/itWaC-cbow.kv"

INPUT_FILES = {
    "train": "data/subset/train/train_subset.csv",
    "validation": "data/subset/train/validation_subset.csv",
    "test": "data/subset/test/labeled_test_subset.csv",
}

OUTPUT_FILES = {
    "train": "itWaC-cbow_train.csv",
    "validation": "itWaC-cbow_validation.csv",
    "test": "itWaC-cbow_test.csv",
}

OUTPUT_DIR = Path("data/features/embedding/non_contestualizzato/itWaC-cbow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================== CONVERSIONE SQLITE → MODELLO GENSIM =====================

def sqlite_to_keyedvectors(sqlite_path, model_path):
    # Se il modello gensim esiste già, lo carica direttamente
    if Path(model_path).exists():
        print(f"Modello già presente: {model_path}")
        return KeyedVectors.load(model_path)

    # Controlla che il file sqlite esista
    if not Path(sqlite_path).exists():
        raise FileNotFoundError(f"File SQLite non trovato: {Path(sqlite_path).resolve()}")

    # Connessione al database sqlite
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Recupera automaticamente la prima tabella del database
    table = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()[0]

    # Recupera i nomi delle colonne
    cols = [r["name"] for r in cur.execute(f'PRAGMA table_info("{table}")')]

    # Colonna della parola
    word_col = [c for c in cols if c.lower() in ["key", "word", "token", "term"]][0]

    # Colonne vettoriali: dim0, dim1, dim2...
    dim_cols = sorted(
        [c for c in cols if c.lower().startswith("dim")],
        key=lambda c: int(c.lower().replace("dim", ""))
    )

    # Query per leggere parole e vettori
    dim_sql = ", ".join([f'"{c}"' for c in dim_cols])
    query = f'SELECT "{word_col}", {dim_sql} FROM "{table}"'

    # Crea il modello gensim vuoto
    model = KeyedVectors(vector_size=len(dim_cols))

    # Legge il database a blocchi
    cur.execute(query)
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break

        words = [r[word_col] for r in rows]
        vectors = np.array([[r[c] for c in dim_cols] for r in rows], dtype=np.float32)

        model.add_vectors(words, vectors)
        print(f"Caricate {len(model)} parole...")

    conn.close()

    # Salva il modello gensim convertito
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print(f"Modello salvato: {model_path}")
    return model


# ===================== TESTO → VETTORE MEDIA + DEVIAZIONE STANDARD =====================

def testo_to_vec(testo, model):
    # Estrae gli embedding delle parole presenti nel vocabolario
    vectors = np.array([
        model[word]
        for word in simple_preprocess(str(testo), deacc=False)
        if word in model
    ])

    # Se nessuna parola è presente nel modello, restituisce un vettore di zeri
    if len(vectors) == 0:
        return np.zeros(model.vector_size * 2)

    # Calcola media e deviazione standard degli embedding delle parole
    mean_vec = vectors.mean(axis=0)
    std_vec = vectors.std(axis=0)

    # Concatena media e deviazione standard in un unico vettore
    return np.concatenate([mean_vec, std_vec])


# ===================== CREAZIONE FILE CSV PER SVM =====================

def salva_embedding(split, input_path, model):
    # Legge il CSV con colonne "text" e "label"
    df = pd.read_csv(input_path)

    # Crea un vettore indipendente per ogni testo
    X = np.vstack([testo_to_vec(testo, model) for testo in df["text"]])

    # Crea la matrice usabile da SVM
    out = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])

    # Associa ogni vettore alla propria label
    out["label"] = df["label"].values

    # Salva il CSV finale
    output_path = OUTPUT_DIR / OUTPUT_FILES[split]
    out.to_csv(output_path, index=False)

    print(f"Salvato: {output_path} | matrice: {X.shape}")


# ===================== MAIN =====================

def main():
    # Carica o converte il modello itWaC-cbow
    model = sqlite_to_keyedvectors(SQLITE_PATH, MODEL_PATH)

    # Ogni file viene processato indipendentemente
    for split, input_path in INPUT_FILES.items():
        salva_embedding(split, input_path, model)


if __name__ == "__main__":
    main()