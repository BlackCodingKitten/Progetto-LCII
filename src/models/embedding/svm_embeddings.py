"""
REMINDER: 
 _____________________________________________________________________________________________________________
| Rappresentazione | Tipo                          | Costruzione vettore del testo                            |
|__________________|_______________________________|__________________________________________________________|
| fastText         | statico, non contestualizzato | media degli embedding delle parole                       |
| itWaC-cbow       | statico, non contestualizzato | media + deviazione standard degli embedding delle parole |
| umBERTo          | contestualizzato              | mean pooling degli hidden states dei token               |
| bge-m3           | contestualizzato              | embedding dense del testo prodotto dal modello           |
|__________________|_______________________________|__________________________________________________________|
"""


from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ===================== CONFIGURAZIONE =====================

RAPPRESENTAZIONI = {
    "bge-m3": {
        "train": "data/features/embedding/contestualizzato/bge-m3/bge-m3_train.csv",
        "validation": "data/features/embedding/contestualizzato/bge-m3/bge-m3_validation.csv",
    },
    "umBERTo": {
        "train": "data/features/embedding/contestualizzato/umBERTo/umBERTo_train.csv",
        "validation": "data/features/embedding/contestualizzato/umBERTo/umBERTo_validation.csv",
    },
    "fastText": {
        "train": "data/features/embedding/non_contestualizzato/fastText_facebook/fastText_train.csv",
        "validation": "data/features/embedding/non_contestualizzato/fastText_facebook/fastText_validation.csv",
    },
    "itWaC-cbow": {
        "train": "data/features/embedding/non_contestualizzato/itWaC-cbow/itWaC-cbow_train.csv",
        "validation": "data/features/embedding/non_contestualizzato/itWaC-cbow/itWaC-cbow_validation.csv",
    },
}

OUTPUT_PATH = Path("results/embedding/embedding_svm_validation_metrics.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ===================== FUNZIONI =====================

def carica_xy(path):
    # Separa embedding e label
    df = pd.read_csv(path)
    return df.drop(columns="label"), df["label"]


def valuta_rappresentazione(nome, paths):
    # Carica train e validation della singola rappresentazione
    X_train, y_train = carica_xy(paths["train"])
    X_val, y_val = carica_xy(paths["validation"])

    # Crea una nuova SVM indipendente per questa rappresentazione
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear")
    )

    # Addestra solo sul train della rappresentazione corrente
    svm.fit(X_train, y_train)

    # Valuta solo sul validation della stessa rappresentazione
    y_pred = svm.predict(X_val)
    y_score = svm.decision_function(X_val)

    return {
        "rappresentazione": nome,
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_score),
    }


# ===================== MAIN =====================

def main():
    # Una SVM separata per ogni tipo di rappresentazione
    risultati = [
        valuta_rappresentazione(nome, paths)
        for nome, paths in RAPPRESENTAZIONI.items()
    ]

    risultati = pd.DataFrame(risultati).sort_values("f1", ascending=False)

    print(risultati)
    risultati.to_csv(OUTPUT_PATH, index=False)

    print(f"\nMetriche salvate in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()