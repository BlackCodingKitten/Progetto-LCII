from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

# ========= PATH =========

# File principale del modello Gensim; il file .vectors.npy deve stare nella stessa cartella.
MODEL_PATH = "data/features/embedding/non_contestualizzato/w2vec/modello/word2vec.wordvectors"

OUT_DIR = Path("data/features/embedding/non_contestualizzato/w2vec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": ("data/subset/train-validation/train_subset.csv", "w2v_zenodo_sif_mean_train.csv"),
    "validation": ("data/subset/train-validation/validation_subset.csv", "w2v_zenodo_sif_mean_validation.csv"),
    "test": ("data/subset/test/labeled_test_subset.csv", "w2v_zenodo_sif_mean_test.csv"),
}

A = 1e-3  # parametro SIF: riduce il peso delle parole troppo frequenti


# ========= FUNZIONI =========

def tokenizza(testo):
    # Tokenizzazione semplice: minuscole + parole alfanumeriche.
    return re.findall(r"\b\w+\b", str(testo).lower())


def leggi_csv(path):
    # Legge text/label e corregge eventuali spazi nei nomi colonna.
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df[["text", "label"]].dropna()


def vettore_documento(testo):
    # Un documento diventa la media pesata SIF dei vettori delle sue parole.
    parole = [w for w in tokenizza(testo) if w in w2v]
    if not parole:
        return np.zeros(DIM)
    pesi = np.array([A / (A + prob.get(w, 0)) for w in parole])
    return np.average(np.vstack([w2v[w] for w in parole]), axis=0, weights=pesi)


# ========= DATI =========

dati = {split: leggi_csv(path) for split, (path, _) in FILES.items()}

# Le frequenze SIF vengono calcolate solo sul training set.
conteggi = Counter(w for testo in dati["train"]["text"] for w in tokenizza(testo))
totale = sum(conteggi.values())
prob = {w: c / totale for w, c in conteggi.items()}


# ========= MODELLO WORD2VEC =========

w2v = KeyedVectors.load(MODEL_PATH, mmap="r")
DIM = w2v.vector_size


# ========= FEATURE =========

# Una riga del CSV resta un documento indipendente.
X = {split: np.vstack(dati[split]["text"].map(vettore_documento)) for split in FILES}

# Rimozione della prima componente principale, stimata solo sul training set.
pc = TruncatedSVD(n_components=1, random_state=42).fit(X["train"]).components_[0]
X = {split: m - m.dot(pc).reshape(-1, 1) * pc for split, m in X.items()}


# ========= SALVATAGGIO =========

for split, (_, nome_output) in FILES.items():
    out = pd.DataFrame(X[split], columns=[f"f_{i}" for i in range(DIM)])
    out["label"] = dati[split]["label"].astype(int).values
    out.to_csv(OUT_DIR / nome_output, index=False)

print(f"File salvati in: {OUT_DIR}")
