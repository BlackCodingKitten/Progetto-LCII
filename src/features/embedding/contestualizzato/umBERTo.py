from pathlib import Path
import numpy as np
import pandas as pd
import torch

from transformers import AutoModel
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"

INPUT_FILES = {
    "train": "data/subset/train/train_subset.csv",
    "validation": "data/subset/train/validation_subset.csv",
    "test": "data/subset/test/labeled_test_subset.csv",
}

OUTPUT_FILES = {
    "train": "umBERTo_train.csv",
    "validation": "umBERTO_validation.csv",
    "test": "umBERTo_test.csv",
}

OUTPUT_DIR = Path("data/features/embedding/contestualizzato/umBERTo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def carica_tokenizer():
    # Scarica tokenizer.json e lo carica direttamente, evitando AutoTokenizer/CamembertTokenizer
    tokenizer_path = hf_hub_download(MODEL_NAME, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=512)
    return tokenizer


def mean_pooling(hidden_state, attention_mask):
    # Media dei token ignorando il padding
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def testo_to_vec(testo, tokenizer, model):
    # Tokenizza un singolo testo
    enc = tokenizer.encode(str(testo))

    input_ids = torch.tensor([enc.ids], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([enc.attention_mask], dtype=torch.long).to(DEVICE)

    # Estrae gli hidden states
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Produce un solo vettore per testo
    vec = mean_pooling(output.last_hidden_state, attention_mask)

    # Normalizza il vettore
    vec = torch.nn.functional.normalize(vec, p=2, dim=1)

    return vec[0].cpu().numpy()


def salva_embedding(split, input_path, tokenizer, model):
    # Legge il CSV con colonne "text" e "label"
    df = pd.read_csv(input_path)

    # Calcola un embedding indipendente per ogni testo
    X = np.vstack([testo_to_vec(testo, tokenizer, model) for testo in df["text"]])

    # Crea matrice usabile da SVM
    out = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])

    # Associa ogni vettore alla propria label
    out["label"] = df["label"].values

    # Salva il CSV finale
    output_path = OUTPUT_DIR / OUTPUT_FILES[split]
    out.to_csv(output_path, index=False)

    print(f"Salvato: {output_path} | matrice: {X.shape}")


def main():
    # Carica tokenizer e modello
    tokenizer = carica_tokenizer()
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # Processa train, validation e test
    for split, input_path in INPUT_FILES.items():
        salva_embedding(split, input_path, tokenizer, model)


if __name__ == "__main__":
    main()