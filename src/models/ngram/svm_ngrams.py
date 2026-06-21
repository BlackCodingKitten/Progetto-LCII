from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


DATASET_PATH = Path("data/features/train_ngram_representations.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots_model_by_model"
EXCEL_OUTPUT = RESULTS_DIR / "svc_linear_test_only_all_models_metrics.xlsx"
CSV_OUTPUT = RESULTS_DIR / "svc_linear_test_only_all_models_metrics.csv"
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
METRICS = {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}


# =========================
# Da JSON n-grammi a dizionari di feature
# =========================
def json_cell_to_items(cell):
    """Converte una cella JSON in lista; celle vuote/NaN diventano liste vuote."""
    return json.loads(cell) if isinstance(cell, str) and cell else []


def rows_to_feature_dicts(X, columns):
    """
    Trasforma ogni riga del DataFrame in un dizionario numerico.
    DictVectorizer convertirà poi questi dizionari in una matrice sparsa sklearn.
    """
    return [
        {
            f"{col}::{item['ngram']}": float(item["freq"])
            for col in columns
            for item in json_cell_to_items(row[col])
        }
        for _, row in X.iterrows()
    ]


# =============== Dataset e rappresentazioni ==========
def load_dataset(path):
    """Carica il CSV e separa feature X e target Y."""
    df = pd.read_csv(path)
    if "label" not in df:
        raise ValueError(f"Manca la colonna 'label' nel file: {path}")
    return df.drop(columns="label"), df["label"].astype(int)


def representation(name, columns):
    """Piccolo helper per rendere leggibile la lista delle rappresentazioni."""
    return {"name": name, "columns": columns}


def get_ngram_representations():
    """Restituisce le stesse rappresentazioni valutate nel codice originale."""
    reps = [representation(f"char_{n}grams", [f"char_{n}grams"]) for n in range(2, 5)]
    reps += [representation("char_2_3_4grams", [f"char_{n}grams" for n in range(2, 5)])]

    for prefix in ["word", "lemma", "pos"]:
        reps += [representation(f"{prefix}_{n}grams", [f"{prefix}_{n}grams"]) for n in range(1, 5)]
        reps += [representation(f"{prefix}_1_2_3_4grams", [f"{prefix}_{n}grams" for n in range(1, 5)])]

    all_columns = [f"char_{n}grams" for n in range(2, 5)] + [
        f"{prefix}_{n}grams" for prefix in ["word", "lemma", "pos"] for n in range(1, 5)
    ]
    return reps + [representation("all_ngrams", all_columns)]


# ======= Modello =========
def build_pipeline(columns):
    """
    Pipeline sklearn compatta:
    1) FunctionTransformer: converte JSON -> lista di dizionari;
    2) DictVectorizer: dizionari -> matrice sparsa;
    3) StandardScaler(with_mean=False): scaling compatibile con matrici sparse;
    4) SVC lineare.
    """
    return Pipeline([
        ("json_to_dicts", FunctionTransformer(rows_to_feature_dicts, kw_args={"columns": columns}, validate=False)),
        ("vectorizer", DictVectorizer(sparse=True)),
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", SVC(kernel="linear")),
    ])


# ====  Output grafici e tabelle =====
def fold_values(row, metric):
    """Estrae i valori dei 5 fold per una metrica da cv_results_."""
    return [row[f"split{i}_test_{metric}"] for i in range(5)]


def plot_model(model_id, representation_name, c_value, row):
    """Salva il grafico Accuracy/F1/ROC-AUC sui 5 fold per un singolo modello."""
    folds = range(1, 6)
    plt.figure(figsize=(10, 6))
    for metric, label in [("accuracy", "Accuracy"), ("f1_macro", "F1 macro"), ("roc_auc", "ROC-AUC")]:
        plt.plot(folds, fold_values(row, metric), marker="o", linestyle="-", label=label)

    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(list(folds))
    plt.title(f"{model_id} - {representation_name} - SVC linear - C={c_value}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    filename = f"{model_id}_{representation_name}_C_{c_value}.png".replace("/", "_").replace(" ", "_")
    output_path = PLOTS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return str(output_path)


def result_row(model_id, representation_name, columns, c_value, row, plot_file):
    """Costruisce una riga della tabella finale, includendo medie/std e valori per fold."""
    base = {
        "model_id": model_id,
        "classifier": "SVC(kernel='linear')",
        "representation": representation_name,
        "columns": ", ".join(columns),
        "C": c_value,
    }
    metrics = {
        f"{metric}_{stat}": row[f"{stat}_test_{metric}"]
        for metric in METRICS
        for stat in ["mean", "std"]
    }
    folds = {
        f"{metric}_fold_{i + 1}": row[f"split{i}_test_{metric}"]
        for metric in METRICS
        for i in range(5)
    }
    return {**base, **metrics, **folds, "plot_file": plot_file}


def save_outputs(df):
    """Salva CSV, Excel e legenda dei campi principali."""
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")

    legend = pd.DataFrame([
        ("model_id", "Identificativo progressivo del modello valutato"),
        ("representation", "Tipo di rappresentazione n-grammi usata"),
        ("C", "Parametro di regolarizzazione di SVC(kernel='linear')"),
        ("accuracy_mean", "Accuracy media sui 5 fold"),
        ("accuracy_std", "Deviazione standard dell'Accuracy sui 5 fold"),
        ("f1_macro_mean", "F1 macro media sui 5 fold"),
        ("f1_macro_std", "Deviazione standard della F1 macro sui 5 fold"),
        ("roc_auc_mean", "ROC-AUC media sui 5 fold"),
        ("roc_auc_std", "Deviazione standard della ROC-AUC sui 5 fold"),
        ("plot_file", "Percorso del plot associato al modello"),
    ], columns=["campo", "significato"])

    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_models", index=False)
        legend.to_excel(writer, sheet_name="legend", index=False)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(DATASET_PATH)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, model_counter = [], 1

    # print("\n===== CROSS-VALIDATION Training Set =====")
    # print(f"Dataset usato: {DATASET_PATH}")

    for rep in get_ngram_representations():
        # print(f"\nRappresentazione: {rep['name']} | Classificatore: SVC(kernel='linear')")

        search = GridSearchCV(
            build_pipeline(rep["columns"]),
            {"svm__C": C_VALUES},
            scoring=METRICS,
            refit=False,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            verbose=1,
            error_score="raise",
        ).fit(X, y)

        results = pd.DataFrame(search.cv_results_).assign(C=lambda d: d["param_svm__C"].astype(float)).sort_values("C")

        for _, row in results.iterrows():
            model_id = f"model_{model_counter}"
            model_counter += 1
            c_value = float(row["C"])
            plot_file = plot_model(model_id, rep["name"], c_value, row)
            rows.append(result_row(model_id, rep["name"], rep["columns"], c_value, row, plot_file))

    df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    save_outputs(df)

    # print(f"DEBUG: Tabella Excel salvata in: {EXCEL_OUTPUT}")
    # print(f"DEBUG: Tabella CSV salvata in: {CSV_OUTPUT}")
    # print(f"DEBUG: Plot salvati in: {PLOTS_DIR}")
    # print("\nDEBUG: Primi modelli ordinati per F1 macro media:")
    # print(df[["model_id", "representation", "C", "accuracy_mean", "f1_macro_mean", "roc_auc_mean", "plot_file"]].head(20))


if __name__ == "__main__":
    main()