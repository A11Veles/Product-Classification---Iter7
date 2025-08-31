import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.svm import LinearSVC
from scipy.sparse import diags

# === CONFIG ===
INPUT_CSV = "data/combined_data.csv"   # your dataset
TEXT_COLS = ["Name", "Description"]         # concatenate into one text field
LEVEL_COLS = ["SegmentTitle", "FamilyTitle", "ClassTitle", "BrickTitle"]
MODEL_DIR = "data/models_icftdcb_svm_full"

os.makedirs(MODEL_DIR, exist_ok=True)

def prep_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .astype(str)
         .str.lower()
         .str.replace(r"\s+"," ", regex=True)
         .str.strip()
    )

@dataclass
class IcfTdcbConfig:
    ngram_range: tuple = (1,2)
    min_df: int = 1
    l2_normalize: bool = True

class IcfTdcbVectorizer:
    def __init__(self, cfg: IcfTdcbConfig = IcfTdcbConfig()):
        self.cfg = cfg
        self.vectorizer: Optional[CountVectorizer] = None
        self.G: Optional[np.ndarray] = None

    def fit(self, X_text: pd.Series, y: pd.Series):
        self.vectorizer = CountVectorizer(ngram_range=self.cfg.ngram_range, min_df=self.cfg.min_df)
        X = self.vectorizer.fit_transform(prep_text(X_text))

        y = pd.Series(y).reset_index(drop=True)
        classes = pd.Categorical(y)
        cats = list(classes.categories)
        C = len(cats); V = X.shape[1]

        tf_tc = np.zeros((C, V), dtype=np.float64)
        for ci, cls in enumerate(cats):
            mask = (classes == cls)         # <-- FIXED: no .to_numpy()
            if np.any(mask):
                tf_tc[ci] = X[mask].sum(axis=0).A1

        # ICF
        cf = (tf_tc > 0).sum(axis=0) + 1e-9
        ICF = np.log(C / cf)

        # TDCB
        col_sum = tf_tc.sum(axis=0) + 1e-9
        P = tf_tc / col_sum
        TDCB = 1.0 - (P * P).sum(axis=0)

        G = ICF * TDCB
        G[G <= 0] = 1e-9
        self.G = G
        return self

    def transform(self, X_text: pd.Series):
        X = self.vectorizer.transform(prep_text(X_text))
        Xw = X @ diags(self.G)
        if self.cfg.l2_normalize:
            Xw = normalize(Xw, norm="l2", axis=1)
        return Xw

    def fit_transform(self, X_text: pd.Series, y: pd.Series):
        self.fit(X_text, y)
        return self.transform(X_text)

def train_full_level(df: pd.DataFrame, text_col: str, label_col: str, C: float = 1.0):
    data = df.dropna(subset=[label_col]).copy()
    if data.empty:
        print(f"[{label_col}] No data — skipped.")
        return None

    le = LabelEncoder()
    y = le.fit_transform(data[label_col].astype(str))

    vec = IcfTdcbVectorizer(IcfTdcbConfig())
    X = vec.fit_transform(data[text_col], pd.Series(y))

    clf = LinearSVC(C=C)
    clf.fit(X, y)

    return {"classifier": clf, "vectorizer": vec, "label_encoder": le}

def save_artifacts(artifacts: dict, level: str):
    out_dir = os.path.join(MODEL_DIR, level)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifacts["classifier"], os.path.join(out_dir, "clf.joblib"))
    joblib.dump(artifacts["label_encoder"], os.path.join(out_dir, "label_encoder.joblib"))
    vstate = {
        "vocabulary_": artifacts["vectorizer"].vectorizer.vocabulary_,
        "ngram_range": artifacts["vectorizer"].vectorizer.ngram_range,
        "G": artifacts["vectorizer"].G,
        "l2_normalize": artifacts["vectorizer"].cfg.l2_normalize,
    }
    joblib.dump(vstate, os.path.join(out_dir, "vectorizer_state.joblib"))
    print(f"Saved model for {level} → {out_dir}")

def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["text"] = (df[TEXT_COLS[0]].fillna("") + " " + df[TEXT_COLS[1]].fillna("")).str.strip()

    for level in LEVEL_COLS:
        if level not in df.columns:
            print(f"[{level}] missing — skipping.")
            continue
        print(f"\n=== Training FULL dataset at {level} level ===")
        artifacts = train_full_level(df, text_col="text", label_col=level, C=1.0)
        if artifacts:
            save_artifacts(artifacts, level)

if __name__ == "__main__":
    main()