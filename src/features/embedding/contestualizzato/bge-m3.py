from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

INPUT_FILES = {
    "train": "data/subset/train/train_subset.csv",
    "validation": "data/subset/train/validation_subset.csv",
    "test": "data/subset/test/labeled_test_subset.csv",
}

OUTPUT_DIR = Path("data/features/embedding/contestualizzato/bge-m3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def salva_embedding(split, input_path, model):
    # Legge il CSV con colonne "text" e "label"
    df = pd.read_csv(input_path)

    # Calcola un embedding indipendente per ogni testo
    embeddings = [
        model.encode(str(testo), normalize_embeddings=True)
        for testo in df["text"]
    ]

    # Trasforma la lista di embedding in una matrice numerica: righe = testi, colonne = dimensioni embedding
    X = np.vstack(embeddings)

    # Crea il dataframe finale utilizzabile da SVM
    out = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])

    # Associa a ogni embedding la label del testo corrispondente
    out["label"] = df["label"].values

    # Salva il CSV finale
    output_path = OUTPUT_DIR / f"bge-m3_{split}.csv"
    out.to_csv(output_path, index=False)

    print(f"Salvato: {output_path} | matrice: {X.shape}")


def main():
    # Carica il modello BGE-M3
    model = SentenceTransformer(MODEL_NAME)

    # Processa separatamente train, validation e test
    for split, input_path in INPUT_FILES.items():
        salva_embedding(split, input_path, model)


if __name__ == "__main__":
    main()