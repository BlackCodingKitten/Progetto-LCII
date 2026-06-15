import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET_PATH = "data/features/train_ngram_representations.csv"

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_model_by_model")

EXCEL_OUTPUT = os.path.join(
    RESULTS_DIR,
    "svc_linear_test_only_all_models_metrics.xlsx"
)

CSV_OUTPUT = os.path.join(
    RESULTS_DIR,
    "svc_linear_test_only_all_models_metrics.csv"
)


# ============================================================
# VETTORIZZATORE JSON -> MATRICE NUMERICA
# ============================================================

class NgramJSONVectorizer(BaseEstimator, TransformerMixin):
    """
    Converte le colonne JSON degli n-grammi in una matrice numerica.

    Ogni cella degli n-grammi contiene una lista JSON del tipo:

    [
        {"ngram": "gatto", "freq": 0.333},
        {"ngram": "mangia", "freq": 0.333}
    ]

    Questa classe trasforma ogni documento in un dizionario:

    {
        "word_1grams::gatto": 0.333,
        "word_1grams::mangia": 0.333
    }

    Poi DictVectorizer converte questi dizionari in una matrice sparsa.
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
            feature_dict = self._row_to_dict(row)
            feature_dicts.append(feature_dict)

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

def load_dataset(path):
    dataframe = pd.read_csv(path)

    if "label" not in dataframe.columns:
        raise ValueError(f"Manca la colonna 'label' nel file: {path}")

    X = dataframe.drop(columns=["label"])
    y = dataframe["label"].astype(int)

    return X, y


# ============================================================
# RAPPRESENTAZIONI N-GRAMMI
# ============================================================

def get_ngram_representations():
    return [
        {
            "name": "char_2grams",
            "columns": ["char_2grams"]
        },
        {
            "name": "char_3grams",
            "columns": ["char_3grams"]
        },
        {
            "name": "char_4grams",
            "columns": ["char_4grams"]
        },
        {
            "name": "char_2_3_4grams",
            "columns": ["char_2grams", "char_3grams", "char_4grams"]
        },

        {
            "name": "word_1grams",
            "columns": ["word_1grams"]
        },
        {
            "name": "word_2grams",
            "columns": ["word_2grams"]
        },
        {
            "name": "word_3grams",
            "columns": ["word_3grams"]
        },
        {
            "name": "word_4grams",
            "columns": ["word_4grams"]
        },
        {
            "name": "word_1_2_3_4grams",
            "columns": [
                "word_1grams",
                "word_2grams",
                "word_3grams",
                "word_4grams"
            ]
        },

        {
            "name": "lemma_1grams",
            "columns": ["lemma_1grams"]
        },
        {
            "name": "lemma_2grams",
            "columns": ["lemma_2grams"]
        },
        {
            "name": "lemma_3grams",
            "columns": ["lemma_3grams"]
        },
        {
            "name": "lemma_4grams",
            "columns": ["lemma_4grams"]
        },
        {
            "name": "lemma_1_2_3_4grams",
            "columns": [
                "lemma_1grams",
                "lemma_2grams",
                "lemma_3grams",
                "lemma_4grams"
            ]
        },

        {
            "name": "pos_1grams",
            "columns": ["pos_1grams"]
        },
        {
            "name": "pos_2grams",
            "columns": ["pos_2grams"]
        },
        {
            "name": "pos_3grams",
            "columns": ["pos_3grams"]
        },
        {
            "name": "pos_4grams",
            "columns": ["pos_4grams"]
        },
        {
            "name": "pos_1_2_3_4grams",
            "columns": [
                "pos_1grams",
                "pos_2grams",
                "pos_3grams",
                "pos_4grams"
            ]
        },

        {
            "name": "all_ngrams",
            "columns": [
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
                "pos_4grams"
            ]
        }
    ]


# ============================================================
# PIPELINE SVC LINEARE
# ============================================================

def build_pipeline(ngram_columns):
    """
    Pipeline:

    1. NgramJSONVectorizer
    2. StandardScaler
    3. SVC(kernel='linear')

    StandardScaler usa with_mean=False perché la matrice è sparsa.
    """

    return Pipeline([
        (
            "vectorizer",
            NgramJSONVectorizer(ngram_columns=ngram_columns)
        ),
        (
            "scaler",
            StandardScaler(with_mean=False)
        ),
        (
            "svm",
            SVC(kernel="linear")
        )
    ])


# ============================================================
# PLOT PER OGNI MODELLO
# ============================================================

def plot_single_model_metrics(
    model_id,
    representation_name,
    c_value,
    fold_accuracy,
    fold_f1_macro,
    fold_roc_auc,
    output_dir
):
    """
    Genera un plot per un singolo modello.

    Il plot mostra le tre metriche nei 5 fold:

    - Accuracy
    - F1 macro
    - ROC-AUC
    """

    os.makedirs(output_dir, exist_ok=True)

    folds = [1, 2, 3, 4, 5]

    plt.figure(figsize=(10, 6))

    plt.plot(
        folds,
        fold_accuracy,
        marker="o",
        linestyle="-",
        label="Accuracy"
    )

    plt.plot(
        folds,
        fold_f1_macro,
        marker="o",
        linestyle="-",
        label="F1 macro"
    )

    plt.plot(
        folds,
        fold_roc_auc,
        marker="o",
        linestyle="-",
        label="ROC-AUC"
    )

    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(folds)

    plt.title(
        f"{model_id} - {representation_name} - SVC linear - C={c_value}"
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    filename = f"{model_id}_{representation_name}_C_{c_value}.png"
    filename = filename.replace("/", "_").replace(" ", "_")

    output_path = os.path.join(output_dir, filename)

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X, y = load_dataset(DATASET_PATH)

    representations = get_ngram_representations()

    c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc"
    }

    all_model_rows = []

    model_counter = 1

    print("\n=====CROSS-VALIDATION Training Set =====")
    print(f"Dataset usato: {DATASET_PATH}")

    for representation in representations:
        representation_name = representation["name"]
        ngram_columns = representation["columns"]

        print("\n======================================")
        print(f"Rappresentazione: {representation_name}")
        print("Classificatore: SVC(kernel='linear')")
        print("======================================")

        pipeline = build_pipeline(ngram_columns)

        param_grid = {
            "svm__C": c_values
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit=False,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            verbose=1,
            error_score="raise"
        )

        grid_search.fit(X, y)

        results = pd.DataFrame(grid_search.cv_results_)

        results["C"] = results["param_svm__C"].astype(float)
        results = results.sort_values(by="C")

        for _, row in results.iterrows():
            model_id = f"model_{model_counter}"
            model_counter += 1

            c_value = float(row["C"])

            fold_accuracy = [
                row["split0_test_accuracy"],
                row["split1_test_accuracy"],
                row["split2_test_accuracy"],
                row["split3_test_accuracy"],
                row["split4_test_accuracy"]
            ]

            fold_f1_macro = [
                row["split0_test_f1_macro"],
                row["split1_test_f1_macro"],
                row["split2_test_f1_macro"],
                row["split3_test_f1_macro"],
                row["split4_test_f1_macro"]
            ]

            fold_roc_auc = [
                row["split0_test_roc_auc"],
                row["split1_test_roc_auc"],
                row["split2_test_roc_auc"],
                row["split3_test_roc_auc"],
                row["split4_test_roc_auc"]
            ]

            plot_file = plot_single_model_metrics(
                model_id=model_id,
                representation_name=representation_name,
                c_value=c_value,
                fold_accuracy=fold_accuracy,
                fold_f1_macro=fold_f1_macro,
                fold_roc_auc=fold_roc_auc,
                output_dir=PLOTS_DIR
            )

            model_row = {
                "model_id": model_id,
                "classifier": "SVC(kernel='linear')",
                "representation": representation_name,
                "columns": ", ".join(ngram_columns),
                "C": c_value,

                "accuracy_mean": row["mean_test_accuracy"],
                "accuracy_std": row["std_test_accuracy"],
                "accuracy_fold_1": row["split0_test_accuracy"],
                "accuracy_fold_2": row["split1_test_accuracy"],
                "accuracy_fold_3": row["split2_test_accuracy"],
                "accuracy_fold_4": row["split3_test_accuracy"],
                "accuracy_fold_5": row["split4_test_accuracy"],

                "f1_macro_mean": row["mean_test_f1_macro"],
                "f1_macro_std": row["std_test_f1_macro"],
                "f1_macro_fold_1": row["split0_test_f1_macro"],
                "f1_macro_fold_2": row["split1_test_f1_macro"],
                "f1_macro_fold_3": row["split2_test_f1_macro"],
                "f1_macro_fold_4": row["split3_test_f1_macro"],
                "f1_macro_fold_5": row["split4_test_f1_macro"],

                "roc_auc_mean": row["mean_test_roc_auc"],
                "roc_auc_std": row["std_test_roc_auc"],
                "roc_auc_fold_1": row["split0_test_roc_auc"],
                "roc_auc_fold_2": row["split1_test_roc_auc"],
                "roc_auc_fold_3": row["split2_test_roc_auc"],
                "roc_auc_fold_4": row["split3_test_roc_auc"],
                "roc_auc_fold_5": row["split4_test_roc_auc"],

                "plot_file": plot_file
            }

            all_model_rows.append(model_row)

    all_models_df = pd.DataFrame(all_model_rows)

    all_models_df = all_models_df.sort_values(
        by="f1_macro_mean",
        ascending=False
    )

    all_models_df.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
        all_models_df.to_excel(
            writer,
            sheet_name="all_models",
            index=False
        )

        explanation_df = pd.DataFrame([
            {
                "campo": "model_id",
                "significato": "Identificativo progressivo del modello valutato"
            },
            {
                "campo": "representation",
                "significato": "Tipo di rappresentazione n-grammi usata"
            },
            {
                "campo": "C",
                "significato": "Parametro di regolarizzazione di SVC(kernel='linear')"
            },
            {
                "campo": "accuracy_mean",
                "significato": "Accuracy media sui 5 fold"
            },
            {
                "campo": "accuracy_std",
                "significato": "Deviazione standard dell'Accuracy sui 5 fold"
            },
            {
                "campo": "f1_macro_mean",
                "significato": "F1 macro media sui 5 fold"
            },
            {
                "campo": "f1_macro_std",
                "significato": "Deviazione standard della F1 macro sui 5 fold"
            },
            {
                "campo": "roc_auc_mean",
                "significato": "ROC-AUC media sui 5 fold"
            },
            {
                "campo": "roc_auc_std",
                "significato": "Deviazione standard della ROC-AUC sui 5 fold"
            },
            {
                "campo": "plot_file",
                "significato": "Percorso del plot associato al modello"
            }
        ])

        explanation_df.to_excel(
            writer,
            sheet_name="legend",
            index=False
        )

    print("\n===== OPERAZIONE COMPLETATA =====")
    print(f"Tabella Excel salvata in: {EXCEL_OUTPUT}")
    print(f"Tabella CSV salvata in: {CSV_OUTPUT}")
    print(f"Plot salvati in: {PLOTS_DIR}")

    print("\nPrimi modelli ordinati per F1 macro media:")
    print(
        all_models_df[
            [
                "model_id",
                "representation",
                "C",
                "accuracy_mean",
                "f1_macro_mean",
                "roc_auc_mean",
                "plot_file"
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()