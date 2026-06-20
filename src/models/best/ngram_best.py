
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# CONFIGURAZIONE
# ============================================================

TRAIN_PATH = Path("data/features/train_ngram_representations.csv")
VALIDATION_PATH = Path("data/features/validation_ngram_representations.csv")
TEST_PATH = Path("data/features/test_ngram_representations.csv")

RESULTS_DIR = Path("results/ngrams/validation-test")
PLOTS_DIR = RESULTS_DIR / "plots_best_model"
MODELS_DIR = RESULTS_DIR / "saved_models"

OUTPUT_METRICS_CSV = RESULTS_DIR / "svc_linear_word1_best_model_train_validation_test_metrics.csv"
OUTPUT_METRICS_XLSX = RESULTS_DIR / "svc_linear_word1_best_model_train_validation_test_metrics.xlsx"
OUTPUT_MODEL = MODELS_DIR / "svc_linear_word1_C_0_001_final_model.joblib"
OUTPUT_METRICS_PLOT = PLOTS_DIR / "svc_linear_word1_train_validation_test_metrics.png"
OUTPUT_LOSS_PLOT = PLOTS_DIR / "svc_linear_word1_hinge_loss_learning_curve.png"

BEST_REPRESENTATION_NAME = "word_1grams"
BEST_NGRAM_COLUMNS = ["word_1grams"]
BEST_C = 0.001


# ============================================================
# LETTURA JSON E VETTORIZZAZIONE
# ============================================================

def parse_cell(cell):
    """Converte una cella JSON in lista; celle vuote o NaN diventano liste vuote."""
    return [] if pd.isna(cell) or cell == "" else json.loads(cell)


def dataframe_to_feature_dicts(X, columns):
    """
    Converte ogni riga del DataFrame in un dizionario di feature.

    Esempio:
    [{"ngram": "cane", "freq": 2}]
    diventa:
    {"word_1grams::cane": 2.0}

    DictVectorizer trasforma poi questi dizionari in una matrice sparsa sklearn.
    """
    return [
        {
            f"{column}::{item['ngram']}": float(item["freq"])
            for column in columns
            for item in parse_cell(row[column])
        }
        for _, row in X.iterrows()
    ]


def build_model():
    """
    Crea la pipeline completa:
    1. FunctionTransformer: JSON -> dizionari di feature;
    2. DictVectorizer: dizionari -> matrice numerica sparsa;
    3. StandardScaler: scaling compatibile con matrici sparse;
    4. SVC lineare con il C migliore già selezionato.
    """
    return make_pipeline(
        FunctionTransformer(
            dataframe_to_feature_dicts,
            validate=False,
            kw_args={"columns": BEST_NGRAM_COLUMNS},
        ),
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        SVC(kernel="linear", C=BEST_C),
    )


# ============================================================
# DATI, METRICHE E SALVATAGGI
# ============================================================

def load_dataset(path, selected_columns=BEST_NGRAM_COLUMNS):
    """Carica un CSV e restituisce solo le colonne n-grammi selezionate e la label."""
    df = pd.read_csv(path)
    required = set(selected_columns) | {"label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Nel file {path} mancano queste colonne: {sorted(missing)}")

    return df[selected_columns].copy(), df["label"].astype(int)


def evaluate(model, X, y, split):
    """Valuta il modello su uno split e stampa report e matrice di confusione."""
    y_pred = model.predict(X)
    y_score = model.decision_function(X)

    scores = {
        "split": split,
        "model": "SVC(kernel='linear')",
        "representation": BEST_REPRESENTATION_NAME,
        "C": BEST_C,
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y, y_score),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    print(f"\n===== RISULTATI SU {split.upper()} =====")
    print(f"Accuracy: {scores['accuracy']:.4f}")
    print(f"F1 macro: {scores['f1_macro']:.4f}")
    print(f"ROC-AUC:  {scores['roc_auc']:.4f}")
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion matrix:")
    print(np.array(scores["confusion_matrix"]))

    return scores


def save_metrics(results):
    """Salva le metriche in CSV e XLSX."""
    df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8")
    df.to_excel(OUTPUT_METRICS_XLSX, index=False)

    print(f"\nMetriche CSV salvate in: {OUTPUT_METRICS_CSV}")
    print(f"Metriche XLSX salvate in: {OUTPUT_METRICS_XLSX}")

    return df


# ============================================================
# GRAFICI
# ============================================================

def plot_metrics(results_df):
    """Disegna un pannello per train, validation e test con le tre metriche principali."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "f1_macro", "roc_auc"]
    labels = ["Accuracy", "F1 macro", "ROC-AUC"]
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, split in zip(axes, ["train", "validation", "test"]):
        row = results_df.loc[results_df["split"] == split].iloc[0]
        y = row[metrics].to_numpy(dtype=float)

        # Mantiene il grafico originale: punti, linea spezzata e curva quadratica estetica.
        ax.scatter(x, y, color="blue", s=30)
        ax.plot(x, y, color="orange", linewidth=1.5)

        x_dense = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_dense, np.polyval(np.polyfit(x, y, deg=2), x_dense), color="green", linewidth=0.8)

        ax.set_title(f"{split.capitalize()} set")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("SVC(kernel='linear') - word_1grams - C=0.001")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUTPUT_METRICS_PLOT, dpi=300)
    plt.close()

    print(f"Grafico metriche salvato in: {OUTPUT_METRICS_PLOT}")


def binary_hinge_loss(y_true, scores):
    """
    Calcola la hinge loss binaria.
    Come nello script originale: label 0 -> -1, label 1 -> +1.
    """
    y_signed = np.where(np.asarray(y_true) == 1, 1, -1)
    return np.maximum(0, 1 - y_signed * scores).mean()


def subset_or_full(X, y, train_size):
    """Restituisce tutto il training set oppure un sottoinsieme stratificato."""
    if train_size == 1.0:
        return X, y

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    indices, _ = next(splitter.split(X, y))
    return X.iloc[indices], y.iloc[indices]


def plot_hinge_loss_curve(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    SVC non ha una loss epoca-per-epoca.
    Per questo si addestra lo stesso modello su porzioni crescenti del training set
    e si misura la hinge loss su train subset, validation e test.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    losses = {"Training loss": [], "Validation loss": [], "Test loss": []}

    for size in train_sizes:
        print(f"\nCalcolo hinge loss con {int(size * 100)}% del training set...")

        X_sub, y_sub = subset_or_full(X_train, y_train, size)
        model = build_model().fit(X_sub, y_sub)

        losses["Training loss"].append(binary_hinge_loss(y_sub, model.decision_function(X_sub)))
        losses["Validation loss"].append(binary_hinge_loss(y_val, model.decision_function(X_val)))
        losses["Test loss"].append(binary_hinge_loss(y_test, model.decision_function(X_test)))

    x = [int(size * len(X_train)) for size in train_sizes]

    plt.figure(figsize=(10, 6))
    for label, values in losses.items():
        plt.plot(x, values, marker="o", label=label)

    plt.xlabel("Numero di documenti usati per il training")
    plt.ylabel("Hinge loss")
    plt.title("Curva di hinge loss - SVC lineare word_1grams C=0.001")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_LOSS_PLOT, dpi=300)
    plt.close()

    print(f"Grafico hinge loss salvato in: {OUTPUT_LOSS_PLOT}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Esegue l'intera procedura finale: training, valutazione, salvataggi e grafici."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n===== CARICAMENTO DATASET =====")
    X_train, y_train = load_dataset(TRAIN_PATH)
    X_val, y_val = load_dataset(VALIDATION_PATH)
    X_test, y_test = load_dataset(TEST_PATH)

    print(f"Train:      {len(X_train)} documenti")
    print(f"Validation: {len(X_val)} documenti")
    print(f"Test:       {len(X_test)} documenti")

    print("\n===== MODELLO MIGLIORE GIÀ SELEZIONATO =====")
    print(f"Modello: SVC(kernel='linear')")
    print(f"Rappresentazione: {BEST_REPRESENTATION_NAME}")
    print(f"C: {BEST_C}")

    print("\n===== ADDESTRAMENTO SU TUTTO IL TRAINING SET =====")
    model = build_model().fit(X_train, y_train)
    joblib.dump(model, OUTPUT_MODEL)
    print(f"Modello finale salvato in: {OUTPUT_MODEL}")

    print("\n===== VALUTAZIONE MODELLO =====")
    results_df = save_metrics([
        evaluate(model, X_train, y_train, "train"),
        evaluate(model, X_val, y_val, "validation"),
        evaluate(model, X_test, y_test, "test"),
    ])

    print("\n===== GRAFICO METRICHE =====")
    plot_metrics(results_df)

    print("\n===== CURVA DI HINGE LOSS =====")
    plot_hinge_loss_curve(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n===== OPERAZIONE COMPLETATA =====")
    print(f"Metriche CSV: {OUTPUT_METRICS_CSV}")
    print(f"Metriche XLSX: {OUTPUT_METRICS_XLSX}")
    print(f"Grafico metriche: {OUTPUT_METRICS_PLOT}")
    print(f"Grafico hinge loss: {OUTPUT_LOSS_PLOT}")
    print(f"Modello finale: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
