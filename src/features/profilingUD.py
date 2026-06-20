# # ===================== TRASFORMAZIONR FILE PER ESSERE PROCESSATI DA PROFILING UD =============================

# from pathlib import Path
# import pandas as pd
# import zipfile
# import re

# # File CSV di partenza: prima colonna testo, seconda colonna label
# FILES = {
#     "train": r"data/subset/train/train_subset.csv",
#     "valid": r"data/subset/train/validation_subset.csv",
#     "test":  r"data/subset/test/labeled_test_subset.csv",
# }

# # Cartella dove salvare zip e mapping label
# OUT = Path(r"data/features/profilingUD/input_zip")
# OUT.mkdir(parents=True, exist_ok=True)

# # Imposta True se i tuoi CSV hanno intestazione; False se NON hanno intestazione
# HAS_HEADER = False

# # Divide il testo in frasi, una frase per riga
# def split_sentences(text):
#     text = "" if pd.isna(text) else str(text)
#     return "\n".join(s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip())

# # Crea uno zip per train, validation e test
# for split, csv_path in FILES.items():

#     # Legge il CSV: se non hai header usa header=None
#     df = pd.read_csv(csv_path) if HAS_HEADER else pd.read_csv(csv_path, header=None)

#     # Prima colonna = testo, seconda colonna = label
#     texts = df.iloc[:, 0]
#     labels = df.iloc[:, 1]

#     # Path zip finale
#     zip_path = OUT / f"{split}_profilingUD_input.zip"

#     # Mapping filename-label da usare dopo per riattaccare le label
#     mapping = []

#     # Scrive un .txt per ogni documento dentro lo zip
#     with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
#         for i, (text, label) in enumerate(zip(texts, labels)):
#             filename = f"{split}_{i:05d}.txt"
#             z.writestr(filename, split_sentences(text))
#             mapping.append({"filename": filename, "label": label})

#     # Salva il mapping delle label
#     pd.DataFrame(mapping).to_csv(OUT / f"{split}_labels_mapping.csv", index=False)

# print("ZIP e mapping creati in:", OUT)



# ========= RIMAPPATURA LABEL =======================

from pathlib import Path
import pandas as pd

BASE = Path(r"data/features/profilingUD")
MAPS = BASE / "input_zip"

FILES = {
    "train": "ud_train.csv",
    "valid": "ud_valid.csv",
    "test":  "ud_test.csv",
}

for split, filename in FILES.items():

    # CSV ufficiale scaricato da Profiling-UD
    ud = pd.read_csv(BASE / filename)

    # Mapping filename-label creato prima
    labels = pd.read_csv(MAPS / f"{split}_labels_mapping.csv")

    # Se Profiling-UD produce "ilename" invece di "filename", lo corregge
    if "ilename" in ud.columns and "filename" not in ud.columns:
        ud = ud.rename(columns={"ilename": "filename"})

    # Aggiunge la label corretta a ogni documento
    ud = ud.merge(labels[["filename", "label"]], on="filename", how="left")

    # Salva sovrascrivendo il file finale
    ud.to_csv(BASE / filename, index=False)

print("Label aggiunte ai CSV Profiling-UD.")