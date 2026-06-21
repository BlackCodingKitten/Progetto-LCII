from pathlib import Path
import numpy as np
import pandas as pd
import fasttext


MODEL_PATH = "data/features/embedding/non_contestualizzato/fastText_facebook/modello/cc.it.300.bin"

INPUT_FILES = {
    "train": "data/subset/train/train_subset.csv",
    "validation": "data/subset/train/validation_subset.csv",
    "test": "data/subset/test/labeled_test_subset.csv",
}

OUTPUT_FILES = {
    "train": "fastText_train.csv",
    "validation": "fastText_validation.csv",
    "test": "fastText_test.csv",
}

OUTPUT_DIR = Path("data/features/embedding/non_contestualizzato/fastText_facebook")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def salva_embedding(split, input_path, model):
    # Legge il CSV con colonne "text" e "label"
    df = pd.read_csv(input_path)

    # Calcola un vettore indipendente per ogni testo
    X = np.vstack([
        model.get_sentence_vector(str(testo).replace("\n", " "))
        for testo in df["text"]
    ])

    # Crea la matrice usabile da SVM
    out = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])

    # Associa ogni vettore alla propria label
    out["label"] = df["label"].values

    # Salva il CSV finale
    output_path = OUTPUT_DIR / OUTPUT_FILES[split]
    out.to_csv(output_path, index=False)

    print(f"Salvato: {output_path} | matrice: {X.shape}")


def main():
    # Carica il modello fastText non contestualizzato
    model = fasttext.load_model(MODEL_PATH) #'FastText' object

    # Processa train, validation e test
    for split, input_path in INPUT_FILES.items():
        salva_embedding(split, input_path, model)


if __name__ == "__main__":
    main()