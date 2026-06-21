from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



UMBERTO = {
    "train": "data/features/embedding/contestualizzato/umBERTo/umBERTo_train.csv",
    # "validation": "data/features/embedding/contestualizzato/umBERTo/umBERTo_validation.csv",
    "test": "data/features/embedding/contestualizzato/umBERTo/umBERTo_test.csv",
}

OUTPUT_PATH = Path("results/embedding/umberto_svm_test_metrics.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ===================== FUNZIONI =====================

def carica_xy(path):
    # Separa embedding e label
    df = pd.read_csv(path)
    return df.drop(columns="label"), df["label"]


def testa_umberto(paths):
    # Carica train e test di umBERTo
    X_train, y_train = carica_xy(paths["train"])
    X_test, y_test = carica_xy(paths["test"])

    # Addestra la SVM lineare solo sul train
    svm = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    svm.fit(X_train, y_train)

    # Valuta il modello migliore sul test set
    y_pred = svm.predict(X_test)
    y_score = svm.decision_function(X_test)

    return {
        "rappresentazione": "umBERTo",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def main():
    # Test finale solo sul modello migliore scelto in validation: umBERTo
    risultati = pd.DataFrame([testa_umberto(UMBERTO)])

    print(risultati)
    risultati.to_csv(OUTPUT_PATH, index=False)

    print(f"\nMetriche di test salvate in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()