# import pandas as pd
# import numpy as np
# import re, warnings
# from dataclasses import dataclass
# from typing import Optional, Dict, Tuple
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import LabelEncoder, normalize
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from scipy.sparse import diags

# warnings.filterwarnings('ignore')
# hierarchy = ['segment','family','class','brick']

# # ---------------- text prep ----------------
# def preprocess_keep_symbols(text):
#     if pd.isna(text): return ""
#     text = str(text).lower()
#     text = re.sub(r'[^a-z0-9\s\+\-/\.]', ' ', text)
#     return ' '.join(text.split())

# # ---------------- data ----------------
# def load_data():
#     train_df = pd.read_csv('data/correctly_matched_mapped_gpc.csv')
#     test1_df = pd.read_csv('data/product_gpc_mapping.csv')
#     test2_df = pd.read_csv('data/validated_actually_labeled_test_dataset.csv')
#     return train_df, test1_df, test2_df

# def split_data(df, seed=42):
#     if 'segment' in df.columns and df['segment'].nunique() > 1:
#         return train_test_split(df, test_size=0.2, random_state=seed, stratify=df['segment'])
#     return train_test_split(df, test_size=0.2, random_state=seed)

# # ---------------- ICF·TDCB vectorizer ----------------
# @dataclass
# class IcfTdcbConfig:
#     ngram_range: Tuple[int,int] = (1,2)
#     min_df: int = 1
#     l2_normalize: bool = True

# class IcfTdcbVectorizer:
#     """
#     Class-aware weighting:
#       - Build counts with CountVectorizer
#       - Aggregate term counts per class from training data
#       - ICF(t) = log(C / cf(t))
#       - TDCB(t) = 1 - sum_c P(t|c)^2, P(t|c) = tf(t,c)/sum_c tf(t,c)
#       - Global weight G(t) = ICF * TDCB
#     """
#     def __init__(self, cfg: IcfTdcbConfig = IcfTdcbConfig()):
#         self.cfg = cfg
#         self.vectorizer: Optional[CountVectorizer] = None
#         self.G: Optional[np.ndarray] = None

#     def fit(self, X_text: pd.Series, y_labels: pd.Series):
#         X_text = X_text.fillna("").astype(str)
#         self.vectorizer = CountVectorizer(ngram_range=self.cfg.ngram_range, min_df=self.cfg.min_df)
#         X = self.vectorizer.fit_transform(X_text)  # [N,V]

#         y = pd.Series(y_labels).reset_index(drop=True)
#         classes = pd.Categorical(y)
#         cats = list(classes.categories)
#         C, V = len(cats), X.shape[1]

#         # term counts per class
#         tf_tc = np.zeros((C, V), dtype=np.float64)
#         for ci, cls in enumerate(cats):
#             mask = (classes == cls)      # boolean numpy array
#             if np.any(mask):
#                 tf_tc[ci] = X[mask].sum(axis=0).A1

#         # ICF
#         cf = (tf_tc > 0).sum(axis=0) + 1e-9
#         ICF = np.log(C / cf)

#         # TDCB
#         col_sum = tf_tc.sum(axis=0) + 1e-9
#         P = tf_tc / col_sum
#         TDCB = 1.0 - (P * P).sum(axis=0)

#         G = ICF * TDCB
#         G[G <= 0] = 1e-9
#         self.G = G
#         return self

#     def transform(self, X_text: pd.Series):
#         X_text = X_text.fillna("").astype(str)
#         X = self.vectorizer.transform(X_text)
#         Xw = X @ diags(self.G)
#         return normalize(Xw, norm="l2", axis=1) if self.cfg.l2_normalize else Xw

# # ---------------- models per level ----------------
# def model_builder():
#     return LinearSVC(C=1.0, class_weight='balanced')

# def train_per_level_icftdcb(tr_text: pd.Series, y_train_df: pd.DataFrame):
#     """
#     Trains one (vectorizer, classifier, label_encoder) per hierarchy level.
#     Returns dicts keyed by level.
#     """
#     vects: Dict[str, IcfTdcbVectorizer] = {}
#     encs:  Dict[str, LabelEncoder] = {}
#     models: Dict[str, Tuple[str, object]] = {}

#     for lvl in hierarchy:
#         # encode labels
#         le = LabelEncoder()
#         y_enc = le.fit_transform(y_train_df[lvl].astype(str))
#         encs[lvl] = le

#         # fit ICF·TDCB vectorizer for THIS level (uses labels of this level)
#         vec = IcfTdcbVectorizer(IcfTdcbConfig())
#         vec.fit(tr_text, y_train_df[lvl])
#         vects[lvl] = vec

#         # train classifier (or constant if only 1 class)
#         if len(np.unique(y_enc)) < 2:
#             models[lvl] = ('const', int(y_enc[0]))
#         else:
#             X_tr_lvl = vec.transform(tr_text)
#             clf = model_builder()
#             clf.fit(X_tr_lvl, y_enc)
#             models[lvl] = ('svm', clf)

#     return vects, models, encs

# def predict_levels_icftdcb(vects, models, encs, text_series: pd.Series):
#     out = {}
#     for lvl in hierarchy:
#         vec = vects[lvl]
#         kind, obj = models[lvl]
#         if kind == 'const':
#             y_pred_enc = np.full(text_series.shape[0], obj, dtype=int)
#         else:
#             X = vec.transform(text_series)
#             y_pred_enc = obj.predict(X)
#         out[lvl] = encs[lvl].inverse_transform(y_pred_enc)
#     return pd.DataFrame(out)

# def eval_weighted_f1(y_true_df, y_pred_df):
#     return float(np.mean([
#         f1_score(y_true_df[l], y_pred_df[l], average='weighted', zero_division=0)
#         for l in hierarchy
#     ]))

# # ---------------- run ----------------
# def run():
#     # load
#     train_df, test1_df, test2_df = load_data()

#     # preprocess train
#     train_df = train_df.copy()
#     train_df['processed_name'] = train_df['product_name'].apply(preprocess_keep_symbols)
#     tr, va = split_data(train_df, seed=42)

#     # targets
#     y_tr = tr[hierarchy].copy()
#     y_va = va[hierarchy].copy()

#     # train per level (ICF·TDCB)
#     vects, models, encs = train_per_level_icftdcb(tr['processed_name'], y_tr)

#     # validate
#     val_preds = predict_levels_icftdcb(vects, models, encs, va['processed_name'])
#     val_f1 = eval_weighted_f1(y_va, val_preds)

#     # test 1 (map columns to hierarchy names)
#     t1 = test1_df.copy()
#     t1['processed_name'] = t1['Name'].apply(preprocess_keep_symbols)
#     y_t1 = t1[['SegmentTitle','FamilyTitle','ClassTitle','BrickTitle']].copy()
#     y_t1.columns = hierarchy
#     p1 = predict_levels_icftdcb(vects, models, encs, t1['processed_name'])
#     test1_f1 = eval_weighted_f1(y_t1, p1)

#     # test 2 (map columns to hierarchy names)
#     t2 = test2_df.copy()
#     t2['processed_name'] = t2['translated_name'].apply(preprocess_keep_symbols)
#     y_t2 = t2[['predicted_segment','predicted_family','predicted_class','predicted_brick']].copy()
#     y_t2.columns = hierarchy
#     p2 = predict_levels_icftdcb(vects, models, encs, t2['processed_name'])
#     test2_f1 = eval_weighted_f1(y_t2, p2)

#     avg_f1 = (test1_f1 + test2_f1) / 2.0
#     print("\nRESULTS (ICF·TDCB + LinearSVC, (1,2)-grams, keep symbols)")
#     print(f"Val F1:   {val_f1:.4f}")
#     print(f"Test1 F1: {test1_f1:.4f}")
#     print(f"Test2 F1: {test2_f1:.4f}")
#     print(f"Avg F1:   {avg_f1:.4f}")

# if __name__ == "__main__":
#     run()


# # run_icf_tdcb_taxonomy_svm_auto.py
# import pandas as pd
# import numpy as np
# import re, warnings, os
# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, List
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder, normalize
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from scipy.sparse import diags, hstack

# warnings.filterwarnings("ignore")

# # ---- File paths (fixed to your datasets) ----
# TRAIN_CSV = "data/correctly_matched_mapped_gpc.csv"
# TEST1_CSV = "data/product_gpc_mapping.csv"
# TEST2_CSV = "data/validated_actually_labeled_test_dataset.csv"

# # ---- Hierarchy (column names as in your files) ----
# hierarchy = ["segment","family","class","brick"]
# test1_cols = ["SegmentTitle","FamilyTitle","ClassTitle","BrickTitle"]
# test2_cols = ["predicted_segment","predicted_family","predicted_class","predicted_brick"]

# # ---- Toggle char n-grams channel (helps robustness). Keep True by default. ----
# USE_CHAR_NGRAMS = True

# # ---------------- text prep ----------------
# def preprocess_keep_symbols(text: str) -> str:
#     if pd.isna(text): return ""
#     text = str(text).lower()
#     text = re.sub(r"[^a-z0-9\s\+\-/\.]", " ", text)
#     return " ".join(text.split())

# # ---------------- taxonomy ICF·TDCB ----------------
# @dataclass
# class IcfTdcbConfig:
#     ngram_range: Tuple[int,int] = (1,2)
#     min_df: int = 1
#     l2_normalize: bool = True

# class IcfTdcbTaxonomyVectorizer:
#     """
#     Build vocabulary and global weights G from taxonomy texts (one tiny doc per label = the label string),
#     then transform product texts with the same vocab/weights.
#     """
#     def __init__(self, cfg: IcfTdcbConfig = IcfTdcbConfig()):
#         self.cfg = cfg
#         self.vectorizer: Optional[CountVectorizer] = None
#         self.G: Optional[np.ndarray] = None

#     def fit_from_labels(self, labels: List[str]):
#         docs = [preprocess_keep_symbols(lbl) for lbl in labels]
#         self.vectorizer = CountVectorizer(ngram_range=self.cfg.ngram_range, min_df=self.cfg.min_df)
#         X_cls = self.vectorizer.fit_transform(docs)        # [C, V]
#         C, V = X_cls.shape

#         # ICF(t) = log(C / cf(t))
#         cf = (X_cls > 0).sum(axis=0).A1 + 1e-9
#         ICF = np.log(max(C,1) / cf)

#         # TDCB(t) = 1 - sum_c P(t|c)^2
#         col_sum = np.asarray(X_cls.sum(axis=0)).ravel() + 1e-9
#         P = X_cls.multiply(1.0 / col_sum)
#         TDCB = 1.0 - np.asarray(P.power(2).sum(axis=0)).ravel()

#         G = ICF * TDCB
#         G[G <= 0] = 1e-9
#         self.G = G
#         return self

#     def transform_products(self, texts: pd.Series):
#         X = self.vectorizer.transform(texts.fillna("").astype(str))
#         Xw = X @ diags(self.G)
#         return normalize(Xw, norm="l2", axis=1) if self.cfg.l2_normalize else Xw

# # ---------------- helpers ----------------
# def per_level_f1(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
#     scores = {}
#     for l in hierarchy:
#         f1 = f1_score(y_true_df[l], y_pred_df[l], average="weighted", zero_division=0)
#         scores[l] = f1
#         print(f"{l.capitalize():7s} weighted F1: {f1:.4f}")
#     return scores

# def model_builder():
#     return LinearSVC(C=1.0, class_weight="balanced")

# # ---------------- main ----------------
# def main():
#     # 1) Load datasets
#     train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
#     test1_df = pd.read_csv(TEST1_CSV, low_memory=False)
#     test2_df = pd.read_csv(TEST2_CSV, low_memory=False)

#     # 2) Preprocess product name columns
#     train_df = train_df.copy()
#     train_df["processed_name"] = train_df["product_name"].apply(preprocess_keep_symbols)

#     test1_df = test1_df.copy()
#     test1_df["processed_name"] = test1_df["Name"].apply(preprocess_keep_symbols)

#     test2_df = test2_df.copy()
#     test2_df["processed_name"] = test2_df["translated_name"].apply(preprocess_keep_symbols)

#     # 3) Split train into train/val
#     if "segment" in train_df.columns and train_df["segment"].nunique() > 1:
#         tr, va = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["segment"])
#     else:
#         tr, va = train_test_split(train_df, test_size=0.2, random_state=42)

#     y_tr = tr[hierarchy].copy()
#     y_va = va[hierarchy].copy()

#     # Optional char vectorizer (shared across levels; fitted only on train text)
#     if USE_CHAR_NGRAMS:
#         char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_df=0.98, norm="l2", sublinear_tf=True)
#         X_char_tr = char_vec.fit_transform(tr["processed_name"])
#     else:
#         char_vec = None

#     # 4) Train per level with taxonomy-driven ICF·TDCB + (optional) char channel
#     vects: Dict[str, IcfTdcbTaxonomyVectorizer] = {}
#     encs:  Dict[str, LabelEncoder] = {}
#     clfs:  Dict[str, Tuple[str, object]] = {}

#     for lvl in hierarchy:
#         # Labels across all datasets (ensures test-only labels are in vocab/weights)
#         labels = set(tr[lvl].dropna().astype(str)) \
#                | set(va[lvl].dropna().astype(str)) \
#                | set(test1_df[["SegmentTitle","FamilyTitle","ClassTitle","BrickTitle"][hierarchy.index(lvl)]].dropna().astype(str)) \
#                | set(test2_df[["predicted_segment","predicted_family","predicted_class","predicted_brick"][hierarchy.index(lvl)]].dropna().astype(str))

#         # Vectorizer built from label strings
#         vec = IcfTdcbTaxonomyVectorizer(IcfTdcbConfig())
#         vec.fit_from_labels(sorted(labels))
#         vects[lvl] = vec

#         # Encode train labels
#         le = LabelEncoder()
#         y_tr_enc = le.fit_transform(y_tr[lvl].astype(str))
#         encs[lvl] = le

#         # Build train features
#         X_tr_word = vec.transform_products(tr["processed_name"])
#         if USE_CHAR_NGRAMS:
#             X_tr = hstack([X_tr_word, X_char_tr], format="csr")
#         else:
#             X_tr = X_tr_word

#         # Train model or const predictor
#         if len(np.unique(y_tr_enc)) < 2:
#             clfs[lvl] = ("const", int(y_tr_enc[0]))
#         else:
#             clf = model_builder()
#             clf.fit(X_tr, y_tr_enc)
#             clfs[lvl] = ("svm", clf)

#     # 5) Prediction helper
#     def predict_block(df_text: pd.Series) -> pd.DataFrame:
#         out = {}
#         X_char = char_vec.transform(df_text) if USE_CHAR_NGRAMS else None
#         for lvl in hierarchy:
#             vec = vects[lvl]
#             kind, obj = clfs[lvl]
#             le = encs[lvl]
#             if kind == "const":
#                 y_pred_enc = np.full(df_text.shape[0], obj, dtype=int)
#             else:
#                 X_word = vec.transform_products(df_text)
#                 X_all = hstack([X_word, X_char], format="csr") if USE_CHAR_NGRAMS else X_word
#                 y_pred_enc = obj.predict(X_all)
#             out[lvl] = le.inverse_transform(y_pred_enc)
#         return pd.DataFrame(out)

#     # 6) Evaluate: VAL
#     print("\nVAL F1 per level:")
#     val_preds = predict_block(va["processed_name"])
#     val_scores = per_level_f1(y_va, val_preds)
#     print(f"Val mean: {np.mean(list(val_scores.values())):.4f}")

#     # 7) Evaluate: TEST1
#     y_t1 = test1_df[["SegmentTitle","FamilyTitle","ClassTitle","BrickTitle"]].copy()
#     y_t1.columns = hierarchy
#     print("\nTEST1 F1 per level:")
#     t1_preds = predict_block(test1_df["processed_name"])
#     t1_scores = per_level_f1(y_t1, t1_preds)
#     print(f"Test1 mean: {np.mean(list(t1_scores.values())):.4f}")

#     # 8) Evaluate: TEST2
#     y_t2 = test2_df[["predicted_segment","predicted_family","predicted_class","predicted_brick"]].copy()
#     y_t2.columns = hierarchy
#     print("\nTEST2 F1 per level:")
#     t2_preds = predict_block(test2_df["processed_name"])
#     t2_scores = per_level_f1(y_t2, t2_preds)
#     print(f"Test2 mean: {np.mean(list(t2_scores.values())):.4f}")

#     # 9) Summary
#     print("\nRESULTS (taxonomy ICF·TDCB + LinearSVC{}):".format(" + char ngrams" if USE_CHAR_NGRAMS else ""))
#     print(f"Val mean:   {np.mean(list(val_scores.values())):.4f}")
#     print(f"Test1 mean: {np.mean(list(t1_scores.values())):.4f}")
#     print(f"Test2 mean: {np.mean(list(t2_scores.values())):.4f}")
#     print(f"Avg(Test1,Test2): {(np.mean(list(t1_scores.values())) + np.mean(list(t2_scores.values())))/2.0:.4f}")

#     # Save predictions
#     out_dir = "/mnt/data/out_icftdcb_taxonomy"
#     os.makedirs(out_dir, exist_ok=True)
#     val_out = pd.concat([va[["product_name"] + hierarchy].reset_index(drop=True), val_preds], axis=1)
#     t1_out  = pd.concat([test1_df[["Name","SegmentTitle","FamilyTitle","ClassTitle","BrickTitle"]].reset_index(drop=True), t1_preds], axis=1)
#     t1_out.columns = ["Name","SegmentTitle","FamilyTitle","ClassTitle","BrickTitle","pred_segment","pred_family","pred_class","pred_brick"]
#     t2_out  = pd.concat([test2_df[["translated_name","predicted_segment","predicted_family","predicted_class","predicted_brick"]].reset_index(drop=True), t2_preds], axis=1)
#     t2_out.columns = ["translated_name","predicted_segment","predicted_family","predicted_class","predicted_brick","pred_segment","pred_family","pred_class","pred_brick"]

#     val_out.to_csv(os.path.join(out_dir, "val_preds.csv"), index=False)
#     t1_out.to_csv(os.path.join(out_dir, "test1_preds.csv"), index=False)
#     t2_out.to_csv(os.path.join(out_dir, "test2_preds.csv"), index=False)
#     print(f"Saved predictions to: {out_dir}")

# if __name__ == "__main__":
#     main()






# run_icf_tdcb_centroid_auto.py
import pandas as pd
import numpy as np
import re, warnings, os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import diags

warnings.filterwarnings("ignore")

# ---- File paths (fixed to your datasets) ----
TRAIN_CSV = "data/correctly_matched_mapped_gpc.csv"
TEST1_CSV = "data/product_gpc_mapping.csv"
TEST2_CSV = "data/validated_actually_labeled_test_dataset.csv"

# ---- Hierarchy (column names as in your files) ----
hierarchy = ["segment","family","class","brick"]
test1_cols = ["SegmentTitle","FamilyTitle","ClassTitle","BrickTitle"]
test2_cols = ["predicted_segment","predicted_family","predicted_class","predicted_brick"]

# ---------------- text prep ----------------
def preprocess_keep_symbols(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\+\-/\.]", " ", text)
    return " ".join(text.split())

# ---------------- taxonomy ICF·TDCB ----------------
@dataclass
class IcfTdcbConfig:
    ngram_range: Tuple[int,int] = (1,2)
    min_df: int = 1
    l2_normalize: bool = True

class IcfTdcbTaxonomyVectorizer:
    """
    Build vocabulary & global weights G from taxonomy texts (one tiny doc per label = the label string),
    then transform product texts with the same vocab/weights.
    """
    def __init__(self, cfg: IcfTdcbConfig = IcfTdcbConfig()):
        self.cfg = cfg
        self.vectorizer: Optional[CountVectorizer] = None
        self.G: Optional[np.ndarray] = None
        self.labels_: Optional[List[str]] = None
        self.cls_matrix_ = None  # normalized class centroids [C, V]

    def fit_from_labels(self, labels: List[str]):
        self.labels_ = list(sorted(set([str(l) for l in labels if pd.notna(l)])))
        docs = [preprocess_keep_symbols(lbl) for lbl in self.labels_]
        self.vectorizer = CountVectorizer(ngram_range=self.cfg.ngram_range, min_df=self.cfg.min_df)
        X_cls = self.vectorizer.fit_transform(docs)        # [C, V]
        C, V = X_cls.shape

        # ICF(t) = log(C / cf(t))
        cf = (X_cls > 0).sum(axis=0).A1 + 1e-9
        ICF = np.log(max(C,1) / cf)

        # TDCB(t) = 1 - sum_c P(t|c)^2
        col_sum = np.asarray(X_cls.sum(axis=0)).ravel() + 1e-9
        P = X_cls.multiply(1.0 / col_sum)
        TDCB = 1.0 - np.asarray(P.power(2).sum(axis=0)).ravel()

        G = ICF * TDCB
        G[G <= 0] = 1e-9
        self.G = G

        # Precompute normalized class centroids with global weights
        Xw_cls = X_cls @ diags(self.G)
        self.cls_matrix_ = normalize(Xw_cls, norm="l2", axis=1)
        return self

    def transform_products(self, texts: pd.Series):
        X = self.vectorizer.transform(texts.fillna("").astype(str))
        Xw = X @ diags(self.G)
        return normalize(Xw, norm="l2", axis=1) if self.cfg.l2_normalize else Xw

    def predict_labels(self, texts: pd.Series) -> List[str]:
        Xp = self.transform_products(texts)
        S = Xp @ self.cls_matrix_.T  # cosine similarity
        idx = np.asarray(S.argmax(axis=1)).ravel()
        return [self.labels_[i] for i in idx]

# ---------------- helpers ----------------
def per_level_f1(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    scores = {}
    for l in hierarchy:
        f1 = f1_score(y_true_df[l], y_pred_df[l], average="weighted", zero_division=0)
        scores[l] = f1
        print(f"{l.capitalize():7s} weighted F1: {f1:.4f}")
    return scores

# ---------------- main ----------------
def main():
    # 1) Load datasets
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    test1_df = pd.read_csv(TEST1_CSV, low_memory=False)
    test2_df = pd.read_csv(TEST2_CSV, low_memory=False)

    # 2) Preprocess product name columns
    train_df["processed_name"] = train_df["product_name"].apply(preprocess_keep_symbols)
    test1_df["processed_name"] = test1_df["Name"].apply(preprocess_keep_symbols)
    test2_df["processed_name"] = test2_df["translated_name"].apply(preprocess_keep_symbols)

    # 3) Split train into train/val (for reporting only)
    if "segment" in train_df.columns and train_df["segment"].nunique() > 1:
        tr, va = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["segment"])
    else:
        tr, va = train_test_split(train_df, test_size=0.2, random_state=42)

    y_va = va[hierarchy].copy()

    # 4) Build one centroid model per level from ALL labels observed across datasets
    models: Dict[str, IcfTdcbTaxonomyVectorizer] = {}
    for lvl in hierarchy:
        labels = set(tr[lvl].dropna().astype(str)) \
               | set(va[lvl].dropna().astype(str)) \
               | set(test1_df[test1_cols[hierarchy.index(lvl)]].dropna().astype(str)) \
               | set(test2_df[test2_cols[hierarchy.index(lvl)]].dropna().astype(str))

        vec = IcfTdcbTaxonomyVectorizer(IcfTdcbConfig())
        vec.fit_from_labels(sorted(labels))
        models[lvl] = vec

    # 5) Predict helper
    def predict_block(df_text: pd.Series) -> pd.DataFrame:
        out = {}
        for lvl in hierarchy:
            out[lvl] = models[lvl].predict_labels(df_text)
        return pd.DataFrame(out)

    # 6) Evaluate: VAL
    print("\nVAL F1 per level (centroid):")
    val_preds = predict_block(va["processed_name"])
    val_scores = per_level_f1(y_va, val_preds)
    print(f"Val mean: {np.mean(list(val_scores.values())):.4f}")

    # 7) Evaluate: TEST1
    y_t1 = test1_df[test1_cols].copy()
    y_t1.columns = hierarchy
    print("\nTEST1 F1 per level (centroid):")
    t1_preds = predict_block(test1_df["processed_name"])
    t1_scores = per_level_f1(y_t1, t1_preds)
    print(f"Test1 mean: {np.mean(list(t1_scores.values())):.4f}")

    # 8) Evaluate: TEST2
    y_t2 = test2_df[test2_cols].copy()
    y_t2.columns = hierarchy
    print("\nTEST2 F1 per level (centroid):")
    t2_preds = predict_block(test2_df["processed_name"])
    t2_scores = per_level_f1(y_t2, t2_preds)
    print(f"Test2 mean: {np.mean(list(t2_scores.values())):.4f}")

    # 9) Summary
    print("\nRESULTS (taxonomy ICF·TDCB centroid):")
    print(f"Val mean:   {np.mean(list(val_scores.values())):.4f}")
    print(f"Test1 mean: {np.mean(list(t1_scores.values())):.4f}")
    print(f"Test2 mean: {np.mean(list(t2_scores.values())):.4f}")
    print(f"Avg(Test1,Test2): {(np.mean(list(t1_scores.values())) + np.mean(list(t2_scores.values())))/2.0:.4f}")

    # Save predictions
    out_dir = "/mnt/data/out_icftdcb_centroid"
    os.makedirs(out_dir, exist_ok=True)
    val_out = pd.concat([va[["product_name"] + hierarchy].reset_index(drop=True), val_preds], axis=1)
    t1_out  = pd.concat([test1_df[["Name"] + test1_cols].reset_index(drop=True), t1_preds], axis=1)
    t1_out.columns = ["Name","SegmentTitle","FamilyTitle","ClassTitle","BrickTitle","pred_segment","pred_family","pred_class","pred_brick"]
    t2_out  = pd.concat([test2_df[["translated_name"] + test2_cols].reset_index(drop=True), t2_preds], axis=1)
    t2_out.columns = ["translated_name","predicted_segment","predicted_family","predicted_class","predicted_brick","pred_segment","pred_family","pred_class","pred_brick"]

    val_out.to_csv(os.path.join(out_dir, "val_preds.csv"), index=False)
    t1_out.to_csv(os.path.join(out_dir, "test1_preds.csv"), index=False)
    t2_out.to_csv(os.path.join(out_dir, "test2_preds.csv"), index=False)
    print(f"Saved predictions to: {out_dir}")

if __name__ == "__main__":
    main()