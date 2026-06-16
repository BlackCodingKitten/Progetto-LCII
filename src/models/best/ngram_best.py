import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# CONFIGURAZIONE
# ============================================================

TRAIN_PATH = "data/features/train_ngram_representations.csv"
VALIDATION_PATH = "data/features/validation_ngram_representations.csv"
TEST_PATH = "data/features/test_ngram_representations.csv"

RESULTS_DIR = "results/ngrams/validation-test"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_best_model")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")

OUTPUT_METRICS_CSV = os.path.join(
    RESULTS_DIR,
    "svc_linear_word1_best_model_train_validation_test_metrics.csv"
)

OUTPUT_METRICS_XLSX = os.path.join(
    RESULTS_DIR,
    "svc_linear_word1_best_model_train_validation_test_metrics.xlsx"
)

OUTPUT_MODEL = os.path.join(
    MODELS_DIR,
    "svc_linear_word1_C_0_001_final_model.joblib"
)

OUTPUT_METRICS_PLOT = os.path.join(
    PLOTS_DIR,
    "svc_linear_word1_train_validation_test_metrics.png"
)

OUTPUT_LOSS_PLOT = os.path.join(
    PLOTS_DIR,
    "svc_linear_word1_hinge_loss_learning_curve.png"
)


# ============================================================
# MODELLO MIGLIORE GIÀ SELEZIONATO
# ============================================================

BEST_REPRESENTATION_NAME = "word_1grams"
BEST_NGRAM_COLUMNS = ["word_1grams"]
BEST_C = 0.001


# ============================================================
# VETTORIZZATORE JSON -> MATRICE NUMERICA
# ============================================================

class NgramJSONVectorizer(BaseEstimator, TransformerMixin):
    """
    Converte celle JSON contenenti n-grammi in una matrice numerica sparsa.

    Ogni cella deve avere una forma tipo:

    [
        {"ngram": "gatto", "freq": 2},
        {"ngram": "mangia", "freq": 1}
    ]

    La feature diventa:

    word_1grams::gatto = 2
    word_1grams::mangia = 1
    """

    def __init__(self, ngram_columns):
        self.ngram_columns = ngram_columns

    def _parse_json_cell(self, cell):
        if pd.isna(cell):
            return []

        if cell == "":
            return []

        return json.loads(cell)

    def _row_to_dict(self, row):
        features = {}

        for column in self.ngram_columns:
            ngrams = self._parse_json_cell(row[column])

            for item in ngrams:
                feature_name = f"{column}::{item['ngram']}"
                features[feature_name] = float(item["freq"])

        return features

    def _dataframe_to_dicts(self, X):
        feature_dicts = []

        for _, row in X.iterrows():
            feature_dicts.append(self._row_to_dict(row))

        return feature_dicts

    def fit(self, X, y=None):
        self.vectorizer_ = DictVectorizer(sparse=True)

        feature_dicts = self._dataframe_to_dicts(X)

        self.vectorizer_.fit(feature_dicts)

        return self

    def transform(self, X):
        feature_dicts = self._dataframe_to_dicts(X)

        return self.vectorizer_.transform(feature_dicts)

    def get_feature_names_out(self):
        return self.vectorizer_.get_feature_names_out()


# ============================================================
# CARICAMENTO DATASET
# ============================================================

def load_dataset(path, selected_columns):
    """
    Carica un dataset CSV e tiene solo:
    - le colonne selezionate degli n-grammi
    - la label

    Nel nostro caso selected_columns = ["word_1grams"].
    """

    dataframe = pd.read_csv(path)

    if "label" not in dataframe.columns:
        raise ValueError(f"Manca la colonna 'label' nel file: {path}")

    missing_columns = []

    for column in selected_columns:
        if column not in dataframe.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            f"Nel file {path} mancano queste colonne: {missing_columns}"
        )

    X = dataframe[selected_columns].copy()
    y = dataframe["label"].astype(int)

    return X, y


# ============================================================
# COSTRUZIONE PIPELINE DEL MODELLO MIGLIORE
# ============================================================

def build_best_model_pipeline():
    """
    Costruisce il modello migliore emerso dalla selezione precedente:

    - rappresentazione: word_1grams
    - classificatore: SVC lineare
    - C: 0.001
    """

    pipeline = Pipeline([
        (
            "vectorizer",
            NgramJSONVectorizer(
                ngram_columns=BEST_NGRAM_COLUMNS
            )
        ),
        (
            "scaler",
            StandardScaler(with_mean=False)
        ),
        (
            "svm",
            SVC(
                kernel="linear",
                C=BEST_C
            )
        )
    ])

    return pipeline


# ============================================================
# ADDESTRAMENTO MODELLO
# ============================================================

def train_final_model(X_train, y_train):
    """
    Addestra il modello finale su tutto il training set.
    """

    model = build_best_model_pipeline()

    model.fit(X_train, y_train)

    return model


# ============================================================
# VALUTAZIONE MODELLO
# ============================================================

def evaluate_model(model, X, y, split_name):
    """
    Valuta il modello su uno split:
    - train
    - validation
    - test

    Calcola:
    - accuracy
    - f1 macro
    - ROC-AUC
    - classification report
    - confusion matrix
    """

    y_pred = model.predict(X)

    y_score = model.decision_function(X)

    accuracy = accuracy_score(y, y_pred)

    f1_macro = f1_score(
        y,
        y_pred,
        average="macro"
    )

    roc_auc = roc_auc_score(
        y,
        y_score
    )

    report_text = classification_report(
        y,
        y_pred,
        digits=4
    )

    matrix = confusion_matrix(
        y,
        y_pred
    )

    print("\n===================================")
    print(f"RISULTATI SU {split_name.upper()}")
    print("===================================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 macro:  {f1_macro:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification report:")
    print(report_text)

    print("Confusion matrix:")
    print(matrix)

    result = {
        "split": split_name,
        "model": "SVC(kernel='linear')",
        "representation": BEST_REPRESENTATION_NAME,
        "C": BEST_C,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist()
    }

    return result


# ============================================================
# SALVATAGGIO MODELLO
# ============================================================

def save_model(model, output_path):
    """
    Salva il modello finale con joblib.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    joblib.dump(
        model,
        output_path
    )

    print(f"\nModello finale salvato in: {output_path}")


# ============================================================
# SALVATAGGIO METRICHE
# ============================================================

def save_metrics(results, output_csv, output_xlsx):
    """
    Salva le metriche principali in CSV e XLSX.
    """

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    results_df.to_excel(
        output_xlsx,
        index=False
    )

    print(f"\nMetriche CSV salvate in: {output_csv}")
    print(f"Metriche XLSX salvate in: {output_xlsx}")

    return results_df


# ============================================================
# GRAFICO METRICHE TRAIN / VALIDATION / TEST
# ============================================================

def plot_metric_panels(results_df, output_path):
    """
    Grafico simile all'esempio mostrato:
    - un pannello per train
    - un pannello per validation
    - un pannello per test

    In ogni pannello vengono mostrate:
    - accuracy
    - f1 macro
    - ROC-AUC
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        sharey=True
    )

    split_order = [
        "train",
        "validation",
        "test"
    ]

    metric_names = [
        "accuracy",
        "f1_macro",
        "roc_auc"
    ]

    metric_labels = [
        "Accuracy",
        "F1 macro",
        "ROC-AUC"
    ]

    for ax, split_name in zip(axes, split_order):
        row = results_df[
            results_df["split"] == split_name
        ]

        if row.empty:
            raise ValueError(
                f"Nessun risultato trovato per lo split: {split_name}"
            )

        row = row.iloc[0]

        y = [
            row["accuracy"],
            row["f1_macro"],
            row["roc_auc"]
        ]

        x = np.arange(
            len(metric_names)
        )

        ax.scatter(
            x,
            y,
            color="blue",
            s=30
        )

        ax.plot(
            x,
            y,
            color="orange",
            linewidth=1.5
        )

        # Curva solo estetica, perché ci sono appena 3 punti.
        x_dense = np.linspace(
            x.min(),
            x.max(),
            200
        )

        coeffs = np.polyfit(
            x,
            y,
            deg=2
        )

        y_smooth = np.polyval(
            coeffs,
            x_dense
        )

        ax.plot(
            x_dense,
            y_smooth,
            color="green",
            linewidth=0.8
        )

        ax.set_title(
            f"{split_name.capitalize()} set"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            metric_labels,
            rotation=20
        )

        ax.set_ylim(
            0.0,
            1.05
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

    fig.suptitle(
        "SVC(kernel='linear') - word_1grams - C=0.001"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.92]
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Grafico metriche salvato in: {output_path}")


# ============================================================
# HINGE LOSS BINARIA
# ============================================================

def compute_binary_hinge_loss(y_true, decision_scores):
    """
    Calcola la hinge loss binaria.

    La SVM produce decision_scores:
    - score > 0: classe positiva
    - score < 0: classe negativa

    Per la hinge loss servono label in {-1, +1}.
    Qui assumiamo:
    - label 0 -> -1
    - label 1 -> +1
    """

    y_true = np.array(y_true)

    y_signed = np.where(
        y_true == 1,
        1,
        -1
    )

    losses = np.maximum(
        0,
        1 - y_signed * decision_scores
    )

    return np.mean(losses)


# ============================================================
# CURVA DI HINGE LOSS
# ============================================================

def plot_hinge_loss_learning_curve(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    output_path
):
    """
    Costruisce una curva sperimentale di hinge loss.

    Nota:
    SVC non produce una curva di loss epoca-per-epoca.
    Quindi qui si addestra il modello su porzioni crescenti
    del training set e si misura la hinge loss su:

    - training subset
    - validation
    - test
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    train_sizes = [
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    train_losses = []
    validation_losses = []
    test_losses = []

    n_train = len(X_train)

    for train_size in train_sizes:
        print(
            f"\nCalcolo hinge loss con {int(train_size * 100)}% del training set..."
        )

        if train_size < 1.0:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_size,
                random_state=42
            )

            subset_indices, _ = next(
                splitter.split(
                    X_train,
                    y_train
                )
            )

            X_train_subset = X_train.iloc[
                subset_indices
            ]

            y_train_subset = y_train.iloc[
                subset_indices
            ]

        else:
            X_train_subset = X_train
            y_train_subset = y_train

        model = build_best_model_pipeline()

        model.fit(
            X_train_subset,
            y_train_subset
        )

        train_scores = model.decision_function(
            X_train_subset
        )

        validation_scores = model.decision_function(
            X_validation
        )

        test_scores = model.decision_function(
            X_test
        )

        train_loss = compute_binary_hinge_loss(
            y_train_subset,
            train_scores
        )

        validation_loss = compute_binary_hinge_loss(
            y_validation,
            validation_scores
        )

        test_loss = compute_binary_hinge_loss(
            y_test,
            test_scores
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        test_losses.append(test_loss)

    x_labels = [
        int(size * n_train)
        for size in train_sizes
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x_labels,
        train_losses,
        marker="o",
        label="Training loss"
    )

    plt.plot(
        x_labels,
        validation_losses,
        marker="o",
        label="Validation loss"
    )

    plt.plot(
        x_labels,
        test_losses,
        marker="o",
        label="Test loss"
    )

    plt.xlabel(
        "Numero di documenti usati per il training"
    )

    plt.ylabel(
        "Hinge loss"
    )

    plt.title(
        "Curva di hinge loss - SVC lineare word_1grams C=0.001"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Grafico hinge loss salvato in: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    print("\n===== CARICAMENTO DATASET =====")

    X_train, y_train = load_dataset(
        TRAIN_PATH,
        selected_columns=BEST_NGRAM_COLUMNS
    )

    X_validation, y_validation = load_dataset(
        VALIDATION_PATH,
        selected_columns=BEST_NGRAM_COLUMNS
    )

    X_test, y_test = load_dataset(
        TEST_PATH,
        selected_columns=BEST_NGRAM_COLUMNS
    )

    print(f"Train:      {len(X_train)} documenti")
    print(f"Validation: {len(X_validation)} documenti")
    print(f"Test:       {len(X_test)} documenti")

    print("\n===== MODELLO MIGLIORE GIÀ SELEZIONATO =====")
    print("Modello: SVC(kernel='linear')")
    print(f"Rappresentazione: {BEST_REPRESENTATION_NAME}")
    print(f"C: {BEST_C}")

    print("\n===== ADDESTRAMENTO SU TUTTO IL TRAINING SET =====")

    model = train_final_model(
        X_train,
        y_train
    )

    save_model(
        model,
        OUTPUT_MODEL
    )

    print("\n===== VALUTAZIONE MODELLO =====")

    train_results = evaluate_model(
        model=model,
        X=X_train,
        y=y_train,
        split_name="train"
    )

    validation_results = evaluate_model(
        model=model,
        X=X_validation,
        y=y_validation,
        split_name="validation"
    )

    test_results = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        split_name="test"
    )

    results = [
        train_results,
        validation_results,
        test_results
    ]

    results_df = save_metrics(
        results=results,
        output_csv=OUTPUT_METRICS_CSV,
        output_xlsx=OUTPUT_METRICS_XLSX
    )

    print("\n===== GRAFICO METRICHE =====")

    plot_metric_panels(
        results_df=results_df,
        output_path=OUTPUT_METRICS_PLOT
    )

    print("\n===== CURVA DI HINGE LOSS =====")

    plot_hinge_loss_learning_curve(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        output_path=OUTPUT_LOSS_PLOT
    )

    print("\n===== OPERAZIONE COMPLETATA =====")
    print(f"Metriche CSV: {OUTPUT_METRICS_CSV}")
    print(f"Metriche XLSX: {OUTPUT_METRICS_XLSX}")
    print(f"Grafico metriche: {OUTPUT_METRICS_PLOT}")
    print(f"Grafico hinge loss: {OUTPUT_LOSS_PLOT}")
    print(f"Modello finale: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()