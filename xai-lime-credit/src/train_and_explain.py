# -*- coding: utf-8 -*-
"""
Decifrando a Caixa-Preta: LIME no German Credit (UCI)
----------------------------------------------------
Este script baixa o dataset "Statlog (German Credit Data)" da UCI,
treina um modelo (RandomForest) com pré-processamento apropriado,
gera métricas e explicações locais com LIME para alguns clientes
(deferidos e negados). Outputs são salvos em ./outputs.

Como executar (Windows):
1) Instale o Python 3.10+ em https://www.python.org (marque "Add python.exe to PATH").
2) (Opcional) Dentro da pasta do projeto, rode: run.bat
   ou manualmente:
      python -m venv .venv
      .venv\Scripts\activate
      pip install -r requirements.txt
      python src/train_and_explain.py

Autor: você :)
"""
import os
import io
import sys
import json
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from lime.lime_tabular import LimeTabularExplainer
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# 1) Download do dataset (arquivo original categórico "german.data")
# ------------------------------------------------------------------
LEGACY_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
NEW_BASE = "https://archive.ics.uci.edu/static/public/144/"
FILENAME = "german.data"
DOCFILE = "german.doc"

def download(url, dest: Path):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        return False

def ensure_dataset():
    data_path = DATA_DIR / FILENAME
    doc_path = DATA_DIR / DOCFILE
    if not data_path.exists():
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        print(f"[INFO] Baixando dataset para {data_path} ...")
        # tenta legacy
        ok = download(LEGACY_BASE + FILENAME, data_path)
        if not ok:
            # tenta novo caminho
            ok = download(NEW_BASE + FILENAME, data_path)
            if not ok:
                raise RuntimeError("Falha ao baixar german.data da UCI. Verifique sua internet ou baixe manualmente e coloque em ./data/german.data")
    if not doc_path.exists():
        download(LEGACY_BASE + DOCFILE, doc_path) or download(NEW_BASE + DOCFILE, doc_path)
    return data_path

# ------------------------------------------------------
# 2) Carregar e nomear colunas (conforme UCI - 20 attrs)
# ------------------------------------------------------
ATTR_NAMES = [
    "StatusExistingChecking",         # 1 (cat)
    "DurationMonths",                 # 2 (num)
    "CreditHistory",                  # 3 (cat)
    "Purpose",                        # 4 (cat)
    "CreditAmount",                   # 5 (num)
    "SavingsAccountBonds",            # 6 (cat)
    "PresentEmploymentSince",         # 7 (cat ord)
    "InstallmentRatePctIncome",       # 8 (num)
    "PersonalStatusSex",              # 9 (cat)
    "OtherDebtorsGuarantors",         # 10 (cat)
    "PresentResidenceSince",          # 11 (num)
    "Property",                       # 12 (cat)
    "Age",                            # 13 (num)
    "OtherInstallmentPlans",          # 14 (cat)
    "Housing",                        # 15 (cat)
    "ExistingCreditsAtBank",          # 16 (num)
    "Job",                            # 17 (cat ord)
    "PeopleLiable",                   # 18 (num)
    "Telephone",                      # 19 (cat)
    "ForeignWorker",                  # 20 (cat)
]

def load_german_df():
    path = ensure_dataset()
    # O arquivo é separado por espaço em branco, 20 features + 1 target no final
    df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    if df.shape[1] != 21:
        raise ValueError(f"Esperava 21 colunas (20 features + target), mas encontrei {df.shape[1]}")
    df.columns = ATTR_NAMES + ["Target"]
    # Alvo: 1 = Bom, 2 = Ruim (UCI); vamos mapear: 1->0 (Good), 2->1 (Bad)
    df["Target"] = (df["Target"] == 2).astype(int)
    return df

# ---------------------------------------------
# 3) Pré-processamento + Treino de um classificador
# ---------------------------------------------
def build_preprocessor(df: pd.DataFrame):
    numeric_cols = ["DurationMonths","CreditAmount","InstallmentRatePctIncome","PresentResidenceSince","Age","ExistingCreditsAtBank","PeopleLiable"]
    categorical_cols = [c for c in ATTR_NAMES if c not in numeric_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return pre, categorical_cols, numeric_cols

def train_model():
    df = load_german_df()
    pre, cat_cols, num_cols = build_preprocessor(df)

    X = df[ATTR_NAMES].copy()
    y = df["Target"].copy()  # 1=Bad, 0=Good

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Ajusta o pré-processador e transforma
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    # Nomes de features (após OneHot + escala)
    feat_names = pre.get_feature_names_out().tolist()
    # Tira prefixos "cat__" e "num__" só para ficar mais legível
    feat_names = [f.replace("cat__", "").replace("num__", "") for f in feat_names]

    # Modelo
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(X_train_enc, y_train)

    # Avaliação
    y_pred = clf.predict(X_test_enc)
    y_proba = clf.predict_proba(X_test_enc)[:,1]
    auc = roc_auc_score(y_test, y_proba)

    rep = classification_report(y_test, y_pred, target_names=["Good(0)", "Bad(1)"])
    cm = confusion_matrix(y_test, y_pred)

    # Salvar métricas
    metrics_txt = textwrap.dedent(f"""
    === Estatísticas do Modelo (RandomForest) ===
    AUC: {auc:.4f}

    Classification report:
    {rep}

    Confusion Matrix (linhas=verdadeiro, colunas=previsto):
    {cm}
    """).strip()
    (OUTPUT_DIR/"metrics.txt").write_text(metrics_txt, encoding="utf-8")
    print(metrics_txt)

    # Grafico: importância global (feature_importances_)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10,6))
    plt.barh([feat_names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Importância (Gini)")
    plt.title("Top 20 Importâncias Globais - RandomForest")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"graficos_importancias_top20.png", dpi=150)
    plt.close()

    return {
        "df": df,
        "pre": pre,
        "clf": clf,
        "X_test_enc": X_test_enc,
        "y_test": y_test.values,
        "feat_names": feat_names,
        "X_test_raw": X_test.reset_index(drop=True),
    }

# ---------------------------------------------
# 4) Explicações locais com LIME
# ---------------------------------------------
def explain_with_lime(pre, clf, X_train_enc, feat_names, instance_enc, n_features=10, file_prefix="sample"):
    # Explainer sobre os dados *já transformados* (OneHot + num escalado)
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train_enc),
        feature_names=feat_names,
        class_names=["Good","Bad"],
        mode="classification",
        discretize_continuous=True,
        verbose=False
    )

    predict_fn = lambda data: clf.predict_proba(data)

    exp = explainer.explain_instance(
        data_row=np.array(instance_enc),
        predict_fn=predict_fn,
        num_features=n_features,
        top_labels=1
    )

    # Salvar HTML e PNG
    html = exp.as_html()
    html_path = OUTPUT_DIR / f"lime_explanation_{file_prefix}.html"
    html_path.write_text(html, encoding="utf-8")

    fig = exp.as_pyplot_figure(label=1)  # label=1 (classe "Bad")
    fig.set_size_inches(8,5)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"lime_explanation_{file_prefix}.png", dpi=150)
    plt.close(fig)

    return html_path

def main():
    stuff = train_model()
    pre = stuff["pre"]
    clf = stuff["clf"]
    X_test_enc = stuff["X_test_enc"]
    feat_names = stuff["feat_names"]
    X_test_raw = stuff["X_test_raw"]
    y_test = stuff["y_test"]

    # Precisamos do X_train_enc para o LIME (dados de treino transformados)
    # Reajustamos só para obter o array com treino mesmo (já fizemos fit acima).
    # Aqui, por simplicidade, retransformamos X_test junto para formar base do explainer.
    # Alternativamente, você pode guardar X_train_enc na função anterior.
    # Vamos refazer com o conjunto completo X para ter mais dados de referência:
    df = stuff["df"]
    X_all_enc = pre.transform(df[ATTR_NAMES])

    # Escolher três casos para explicar: (i) um caso previsto como "Bad", (ii) um "Good", (iii) o mais arriscado
    preds = clf.predict(X_test_enc)
    prob_bad = clf.predict_proba(X_test_enc)[:,1]

    # (i) primeiro caso previsto como "Bad"
    idx_bad = int(np.where(preds == 1)[0][0]) if np.any(preds==1) else 0
    explain_with_lime(pre, clf, X_all_enc, feat_names, X_test_enc[idx_bad], file_prefix=f"test_bad_idx{idx_bad}")

    # (ii) primeiro caso previsto como "Good"
    idx_good = int(np.where(preds == 0)[0][0]) if np.any(preds==0) else 1
    explain_with_lime(pre, clf, X_all_enc, feat_names, X_test_enc[idx_good], file_prefix=f"test_good_idx{idx_good}")

    # (iii) caso mais arriscado (maior probabilidade de Bad)
    idx_risky = int(np.argmax(prob_bad))
    explain_with_lime(pre, clf, X_all_enc, feat_names, X_test_enc[idx_risky], file_prefix=f"test_toprisk_idx{idx_risky}")

    # Salvar também uma amostra de X_test_raw para facilitar revisão manual
    sample_df = X_test_raw.copy()
    sample_df["y_true"] = y_test
    sample_df["y_pred"] = preds
    sample_df["p_bad"] = prob_bad
    sample_df.head(20).to_csv(OUTPUT_DIR/"amostra_predicoes.csv", index=False, encoding="utf-8")

    print("\n[OK] Arquivos gerados na pasta 'outputs/'. Consulte:")
    print(" - metrics.txt")
    print(" - graficos_importancias_top20.png")
    print(" - lime_explanation_test_bad_*.html/.png")
    print(" - lime_explanation_test_good_*.html/.png")
    print(" - lime_explanation_test_toprisk_*.html/.png")
    print(" - amostra_predicoes.csv")

if __name__ == "__main__":
    main()
