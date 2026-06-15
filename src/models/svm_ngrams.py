import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# VETTORIZZATORE PER LE COLONNE JSON DEGLI N-GRAMMI
# ============================================================

class NgramJSONVectorizer(BaseEstimator, TransformerMixin):
    """
    Trasforma le colonne JSON degli n-grammi in una matrice numerica.

    Il CSV contiene celle del tipo:

    [
        {"ngram": "gatto", "freq": 0.333},
        {"ngram": "mangia", "freq": 0.333}
    ]

    Questa classe le trasforma in dizionari del tipo:

    {
        "word_1grams::gatto": 0.333,
        "word_1grams::mangia": 0.333
    }

    Poi DictVectorizer trasforma questi dizionari in una matrice numerica sparsa.
    """

    def __init__(self, ngram_columns):
        self.ngram_columns = ngram_columns
        self.vectorizer = DictVectorizer(sparse=True)

    def _parse_json_cell(self, cell):
        if pd.isna(cell):
            return []

        if cell == "":
            return []

        return json.loads(cell)

    def _row_to_feature_dict(self, row):
        features = {}

        for column in self.ngram_columns:
            if column not in row:
                raise ValueError(f"Colonna mancante nel dataset: {column}")

            ngram_list = self._parse_json_cell(row[column])

            for item in ngram_list:
                ngram = item["ngram"]
                freq = item["freq"]

                feature_name = f"{column}::{ngram}"
                features[feature_name] = freq

        return features

    def _dataframe_to_feature_dicts(self, X):
        feature_dicts = []

        for _, row in X.iterrows():
            feature_dict = self._row_to_feature_dict(row)
            feature_dicts.append(feature_dict)

        return feature_dicts

    def fit(self, X, y=None):
        feature_dicts = self._dataframe_to_feature_dicts(X)
        self.vectorizer.fit(feature_dicts)
        return self

    def transform(self, X):
        feature_dicts = self._dataframe_to_feature_dicts(X)
        return self.vectorizer.transform(feature_dicts)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


# ============================================================
# CARICAMENTO DEI DATASET
# ============================================================

def load_dataset(path):
    dataframe = pd.read_csv(path)

    if "label" not in dataframe.columns:
        raise ValueError(f"Nel file {path} manca la colonna 'label'.")

    X = dataframe.drop(columns=["label"])
    y = dataframe["label"].astype(int)

    return X, y


# ============================================================
# RAPPRESENTAZIONI N-GRAMMI DA TESTARE
# ============================================================

def get_ngram_representations():
    """
    Ogni configurazione indica quali colonne n-grammi usare.
    """

    representations = [
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
            "name": "word_lemma",
            "columns": [
                "word_1grams",
                "word_2grams",
                "word_3grams",
                "word_4grams",
                "lemma_1grams",
                "lemma_2grams",
                "lemma_3grams",
                "lemma_4grams"
            ]
        },

        {
            "name": "word_pos",
            "columns": [
                "word_1grams",
                "word_2grams",
                "word_3grams",
                "word_4grams",
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

    return representations


# ============================================================
# PIPELINE BASE
# ============================================================

def build_pipeline(ngram_columns):
    """
    Costruisce la pipeline:

    1. NgramJSONVectorizer:
       converte JSON degli n-grammi in dizionari numerici.

    2. StandardScaler:
       scala le feature.
       with_mean=False è obbligatorio perché la matrice è sparsa.

    3. classifier:
       viene sostituito da GridSearchCV con SVC, LinearSVC o SGDClassifier.
    """

    pipeline = Pipeline([
        (
            "vectorizer",
            NgramJSONVectorizer(ngram_columns=ngram_columns)
        ),
        (
            "scaler",
            StandardScaler(with_mean=False)
        ),
        (
            "classifier",
            LinearSVC()
        )
    ])

    return pipeline


# ============================================================
# GRIGLIA DEI MODELLI E DEGLI IPERPARAMETRI
# ============================================================

def get_param_grid():
    """
    GridSearchCV proverà tre famiglie di modelli:

    1. SVC(kernel="linear") con diversi valori di C
    2. LinearSVC con diversi valori di C
    3. SGDClassifier con diversi valori di alpha

    C:
        regolarizzazione per SVC e LinearSVC.
        Valori più piccoli = maggiore regolarizzazione.

    alpha:
        regolarizzazione per SGDClassifier.
        Valori più grandi = maggiore regolarizzazione.
    """

    param_grid = [
        {
            "classifier": [
                SVC(
                    kernel="linear",
                    cache_size=2000
                )
            ],
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]
        },
        {
            "classifier": [
                LinearSVC(
                    max_iter=10000
                )
            ],
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]
        },
        {
            "classifier": [
                SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=42
                )
            ],
            "classifier__alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01]
        }
    ]

    return param_grid


# ============================================================
# NOME DEL CLASSIFICATORE
# ============================================================

def get_classifier_name(classifier):
    return classifier.__class__.__name__


# ============================================================
# PLOT DELLA GRID SEARCH
# ============================================================

def plot_grid_search_results(grid_search, representation_name, plots_dir):
    """
    Salva i plot di stabilità della GridSearchCV.

    Per ogni rappresentazione e per ogni classificatore crea un grafico con:

    - Accuracy media sui 5 fold
    - F1 macro media sui 5 fold
    - ROC-AUC media sui 5 fold
    - area ± deviazione standard

    Per SVC e LinearSVC l'asse X è C.
    Per SGDClassifier l'asse X è alpha.
    """

    os.makedirs(plots_dir, exist_ok=True)

    results = pd.DataFrame(grid_search.cv_results_)

    results["classifier_name"] = results["param_classifier"].apply(
        lambda classifier: classifier.__class__.__name__
    )

    metric_info = [
        {
            "name": "Accuracy",
            "mean_column": "mean_test_accuracy",
            "std_column": "std_test_accuracy"
        },
        {
            "name": "F1 macro",
            "mean_column": "mean_test_f1_macro",
            "std_column": "std_test_f1_macro"
        },
        {
            "name": "ROC-AUC",
            "mean_column": "mean_test_roc_auc",
            "std_column": "std_test_roc_auc"
        }
    ]

    classifier_parameter = {
        "SVC": "param_classifier__C",
        "LinearSVC": "param_classifier__C",
        "SGDClassifier": "param_classifier__alpha"
    }

    for classifier_name, parameter_column in classifier_parameter.items():
        classifier_results = results[
            results["classifier_name"] == classifier_name
        ].copy()

        if classifier_results.empty:
            continue

        classifier_results[parameter_column] = classifier_results[
            parameter_column
        ].astype(float)

        classifier_results = classifier_results.sort_values(
            by=parameter_column
        )

        x_values = classifier_results[parameter_column].values

        plt.figure(figsize=(10, 6))

        for metric in metric_info:
            mean_scores = classifier_results[metric["mean_column"]].values
            std_scores = classifier_results[metric["std_column"]].values

            line, = plt.plot(
                x_values,
                mean_scores,
                marker="o",
                linestyle="-",
                label=metric["name"]
            )

            plt.fill_between(
                x_values,
                mean_scores - std_scores,
                mean_scores + std_scores,
                alpha=0.15,
                color=line.get_color()
            )

        plt.xscale("log")

        if classifier_name in ["SVC", "LinearSVC"]:
            parameter_label = "Parametro C"
        else:
            parameter_label = "Parametro alpha"

        plt.xlabel(f"{parameter_label} - scala logaritmica")
        plt.ylabel("Score medio in cross-validation")
        plt.title(
            f"Stabilità 5-fold CV - {representation_name} - {classifier_name}"
        )
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc="best")
        plt.tight_layout()

        filename = f"{representation_name}_{classifier_name}_gridsearch_stability.png"
        filename = filename.replace("/", "_").replace(" ", "_")

        output_path = os.path.join(plots_dir, filename)

        plt.savefig(output_path, dpi=300)
        plt.close()


# ============================================================
# ESTRAZIONE DEI PUNTEGGI PER ROC-AUC
# ============================================================

def get_scores_for_roc_auc(model, X):
    """
    La ROC-AUC non si calcola sulle predizioni 0/1,
    ma sui punteggi continui del modello.

    I tre modelli usati espongono decision_function().
    """

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        if len(scores.shape) == 2:
            return scores[:, 1]

        return scores

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, 1]

    raise ValueError(
        "Il modello non fornisce né decision_function né predict_proba."
    )


# ============================================================
# VALUTAZIONE SU VALIDATION E TEST
# ============================================================

def evaluate_model(model, X, y, split_name):
    predictions = model.predict(X)
    scores = get_scores_for_roc_auc(model, X)

    accuracy = accuracy_score(y, predictions)
    f1_macro = f1_score(y, predictions, average="macro")
    roc_auc = roc_auc_score(y, scores)

    print(f"\n===== RISULTATI SU {split_name.upper()} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(y, predictions))

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():
    train_path = "data/features/test_ngram_representations.csv"
    # validation_path = "data/ngrams/validation_ngrams.csv"
    # test_path = "data/ngrams/test_ngrams.csv"

    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Caricamento dei tre dataset già trasformati in n-grammi
    X_train, y_train = load_dataset(train_path)
    X_validation, y_validation = load_dataset(validation_path)
    X_test, y_test = load_dataset(test_path)

    # 2. Definizione delle rappresentazioni n-grammi
    representations = get_ngram_representations()

    # 3. Definizione della 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 4. Metriche da calcolare nella GridSearchCV
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc"
    }

    all_results = []

    best_global_score = -1.0
    best_global_representation = None
    best_global_grid_search = None

    print("\n===== MODEL SELECTION CON GRIDSEARCHCV =====")

    # 5. Ciclo sulle rappresentazioni n-grammi
    for representation in representations:
        representation_name = representation["name"]
        ngram_columns = representation["columns"]

        print(f"\n\n===== RAPPRESENTAZIONE: {representation_name} =====")

        pipeline = build_pipeline(ngram_columns=ngram_columns)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=get_param_grid(),
            scoring=scoring,
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            verbose=2
        )

        # 6. GridSearchCV SOLO sul training set
        grid_search.fit(X_train, y_train)

        # 7. Salvataggio dei plot di stabilità
        plot_grid_search_results(
            grid_search=grid_search,
            representation_name=representation_name,
            plots_dir=plots_dir
        )

        # 8. Conversione risultati GridSearchCV in DataFrame
        results = pd.DataFrame(grid_search.cv_results_)

        results["representation"] = representation_name
        results["columns"] = ", ".join(ngram_columns)

        results["classifier_name"] = results["param_classifier"].apply(
            lambda classifier: classifier.__class__.__name__
        )

        # 9. Salvataggio risultati specifici della rappresentazione
        representation_results_path = os.path.join(
            results_dir,
            f"gridsearch_results_{representation_name}.csv"
        )

        results.to_csv(
            representation_results_path,
            index=False,
            encoding="utf-8"
        )

        # 10. Miglior risultato per questa rappresentazione
        best_index = grid_search.best_index_
        best_row = results.iloc[best_index]

        best_representation_score = grid_search.best_score_

        print("\nMiglior modello per questa rappresentazione:")
        print(f"Rappresentazione: {representation_name}")
        print(f"Classificatore: {best_row['classifier_name']}")
        print(f"Parametri: {grid_search.best_params_}")
        print(f"F1 macro media CV: {best_representation_score:.4f}")

        all_results.append({
            "representation": representation_name,
            "columns": ", ".join(ngram_columns),

            "best_classifier": best_row["classifier_name"],
            "best_params": str(grid_search.best_params_),

            "cv_accuracy_mean": best_row["mean_test_accuracy"],
            "cv_accuracy_std": best_row["std_test_accuracy"],

            "cv_f1_macro_mean": best_row["mean_test_f1_macro"],
            "cv_f1_macro_std": best_row["std_test_f1_macro"],

            "cv_roc_auc_mean": best_row["mean_test_roc_auc"],
            "cv_roc_auc_std": best_row["std_test_roc_auc"],

            "representation_results_file": representation_results_path
        })

        # 11. Aggiornamento del miglior modello globale
        if best_representation_score > best_global_score:
            best_global_score = best_representation_score
            best_global_representation = representation_name
            best_global_grid_search = grid_search

    # 12. Salvataggio tabella riassuntiva di tutte le rappresentazioni
    summary_df = pd.DataFrame(all_results)

    summary_df = summary_df.sort_values(
        by="cv_f1_macro_mean",
        ascending=False
    )

    summary_path = os.path.join(
        results_dir,
        "svm_ngram_gridsearch_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    # 13. Miglior modello globale
    best_model = best_global_grid_search.best_estimator_

    print("\n\n===== MIGLIOR MODELLO GLOBALE =====")
    print(f"Rappresentazione: {best_global_representation}")
    print(f"Parametri: {best_global_grid_search.best_params_}")
    print(f"F1 macro media CV: {best_global_score:.4f}")

    # 14. Valutazione finale su validation set
    validation_results = evaluate_model(
        best_model,
        X_validation,
        y_validation,
        "validation"
    )

    # 15. Valutazione finale su test set
    test_results = evaluate_model(
        best_model,
        X_test,
        y_test,
        "test"
    )

    # 16. Salvataggio risultati finali
    final_results = pd.DataFrame([
        {
            "best_representation": best_global_representation,
            "best_params": str(best_global_grid_search.best_params_),
            "selection_metric": "f1_macro",
            "train_cv_f1_macro_mean": best_global_score,

            "validation_accuracy": validation_results["accuracy"],
            "validation_f1_macro": validation_results["f1_macro"],
            "validation_roc_auc": validation_results["roc_auc"],

            "test_accuracy": test_results["accuracy"],
            "test_f1_macro": test_results["f1_macro"],
            "test_roc_auc": test_results["roc_auc"]
        }
    ])

    final_results_path = os.path.join(
        results_dir,
        "svm_ngram_best_model_final_results.csv"
    )

    final_results.to_csv(
        final_results_path,
        index=False,
        encoding="utf-8"
    )

    # 17. Salvataggio del modello migliore
    model_path = os.path.join(
        results_dir,
        "svm_ngram_best_model.joblib"
    )

    joblib.dump(best_model, model_path)

    print("\nFile salvati:")
    print(summary_path)
    print(final_results_path)
    print(model_path)
    print(plots_dir)


if __name__ == "__main__":
    main()