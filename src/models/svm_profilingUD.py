from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA = Path("data/features/profilingUD")
RES = Path("results/profilingUD")
RES.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "ud_train.csv")
valid = pd.read_csv(DATA / "ud_valid.csv")
test  = pd.read_csv(DATA / "ud_test.csv")

# Escludo colonne non numeriche o non utili alla classificazione
exclude = {"filename", "text", "label"}

# Tengo solo feature numeriche comuni a train/valid/test
features = sorted(
    (set(train.columns) & set(valid.columns) & set(test.columns)) - exclude
)

# Converto le feature in numerico e sostituisco eventuali valori mancanti con 0
X_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(0)
X_valid = valid[features].apply(pd.to_numeric, errors="coerce").fillna(0)
X_test  = test[features].apply(pd.to_numeric, errors="coerce").fillna(0)

le = LabelEncoder()
y_train = le.fit_transform(train["label"])
y_valid = le.transform(valid["label"])
y_test  = le.transform(test["label"])

# ========= STANDARDIZZAZIONE =========
# La scala è imparata solo sul training set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

# ========================== MODELLO ============
# SGDClassifier con loss hinge = SVM lineare addestrata con discesa del gradiente
model = SGDClassifier(loss="hinge", alpha=1e-4, random_state=42)

# Funzione per calcolare la hinge loss media
def hinge_loss(y, score):
    y = np.where(y == 1, 1, -1)
    return np.maximum(0, 1 - y * score).mean()

# ================ TRAINING + CURVA DI LOSS ==============
losses = []

for epoch in range(1, 51):

    # Addestramento incrementale epoca per epoca
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))

    # Score continui prodotti dalla SVM
    s_train = model.decision_function(X_train)
    s_valid = model.decision_function(X_valid)
    s_test  = model.decision_function(X_test)

    # Salvo la loss sui tre set
    losses.append({
        "epoch": epoch,
        "train_loss": hinge_loss(y_train, s_train),
        "valid_loss": hinge_loss(y_valid, s_valid),
        "test_loss": hinge_loss(y_test, s_test),
    })

losses = pd.DataFrame(losses)
losses.to_csv(RES / "profilingUD_loss_curve.csv", index=False)

# ================== GRAFICO LOSS ======================
plt.figure()
plt.plot(losses["epoch"], losses["train_loss"], label="Train")
plt.plot(losses["epoch"], losses["valid_loss"], label="Validation")
plt.plot(losses["epoch"], losses["test_loss"], label="Test")
plt.xlabel("Epoca")
plt.ylabel("Hinge loss")
plt.title("Curva di loss - SVM lineare Profiling-UD")
plt.legend()
plt.tight_layout()
plt.savefig(RES / "profilingUD_loss_curve.png", dpi=300)
plt.close()

#============= METRICHE DI VALUTAZIONE ======================
def evaluate(name, X, y):
    pred = model.predict(X)
    score = model.decision_function(X)

    return {
        "set": name,
        "accuracy": accuracy_score(y, pred),
        "f1_score": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, score),
    }, pred, score

valid_metrics, valid_pred, valid_score = evaluate("validation", X_valid, y_valid)
test_metrics, test_pred, test_score = evaluate("test", X_test, y_test)

# Salvo metriche validation/test
metrics = pd.DataFrame([valid_metrics, test_metrics])
metrics.to_csv(RES / "profilingUD_metrics.csv", index=False)

# Salvo report testuale completo
with open(RES / "profilingUD_classification_report.txt", "w", encoding="utf-8") as f:
    f.write("VALIDATION\n")
    f.write(classification_report(y_valid, valid_pred, target_names=le.classes_.astype(str)))
    f.write("\n\nTEST\n")
    f.write(classification_report(y_test, test_pred, target_names=le.classes_.astype(str)))

# ========== 5 DOCUMENTI PIÙ INCERTI SUL TEST =================
# Più lo score è vicino a 0, più il modello è incerto
uncertain = test.copy()
uncertain["prediction"] = le.inverse_transform(test_pred)
uncertain["decision_score"] = test_score
uncertain["uncertainty"] = abs(test_score)

uncertain.sort_values("uncertainty").head(5).to_csv(
    RES / "profilingUD_5_most_uncertain_test_documents.csv",
    index=False
)

# ============== 20 FEATURE PIÙ RILEVANTI ===========
# Nella SVM lineare le feature più importanti sono quelle con peso assoluto maggiore
top20 = pd.DataFrame({
    "feature": features,
    "weight": model.coef_[0],
})

top20["abs_weight"] = top20["weight"].abs()

top20.sort_values("abs_weight", ascending=False).head(20).drop(
    columns="abs_weight"
).to_csv(
    RES / "profilingUD_top20_features.csv",
    index=False
)

# ============ SALVATAGGIO MODELLO ================
joblib.dump(model, RES / "profilingUD_linear_svm.joblib")
joblib.dump(scaler, RES / "profilingUD_scaler.joblib")
joblib.dump(le, RES / "profilingUD_label_encoder.joblib")

##print(metrics)
##print(f"\nFile salvati in: {RES}")