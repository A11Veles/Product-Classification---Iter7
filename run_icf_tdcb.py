import pandas as pd
import numpy as np
from scipy.sparse import diags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from src.modules.db import TeradataDatabase
from teradataml import *


td_db = TeradataDatabase() 
td_db.connect()

# ------------------------
# 1. Minimal helper
# ------------------------
def prep_text(s: pd.Series) -> pd.Series:
    """Simple cleaner: lowercase + trim"""
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

# ------------------------
# 2. Main function
# ------------------------
def icf_tdcb_predict(gpc_df, products_df, k=3,
                     class_id_col="class_id", class_name_col="class_name", class_text_col="class_text",
                     product_id_col="product_id", product_text_col="translated_name"):
    # Clean inputs
    gpc_df = gpc_df.copy()
    gpc_df[class_text_col] = prep_text(gpc_df[class_text_col])
    products_df = products_df.copy()
    products_df[product_text_col] = prep_text(products_df[product_text_col])

    # Vectorize class texts
    vect = CountVectorizer(ngram_range=(1,2), min_df=1)
    X_cls = vect.fit_transform(gpc_df[class_text_col])
    C = X_cls.shape[0]

    # ICF
    cf = (X_cls > 0).sum(axis=0).A1 + 1e-9
    ICF = np.log(C / cf)

    # TDCB
    col_sums = np.asarray(X_cls.sum(axis=0)).ravel() + 1e-9
    P = X_cls.multiply(1.0 / col_sums)
    TDCB = 1.0 - np.asarray(P.power(2).sum(axis=0)).ravel()

    # Global weights
    G = np.maximum(ICF * TDCB, 1e-9)

    # Class centroids
    V_cls = normalize(X_cls @ diags(G), norm="l2", axis=1)

    # Products
    X_prod = vect.transform(products_df[product_text_col])
    V_prod = normalize(X_prod @ diags(G), norm="l2", axis=1)

    # Scores
    S = (V_prod @ V_cls.T).toarray()
    topk_idx = (-S).argsort(1)[:, :k]
    topk_scores = np.take_along_axis(S, topk_idx, axis=1)

    # Build output
    rows = []
    for i in range(S.shape[0]):
        base = {}
        if product_id_col in products_df.columns:
            base["product_id"] = products_df.loc[i, product_id_col]
        base["product_text"] = products_df.loc[i, product_text_col]
        for j in range(k):
            base[f"class_{j+1}_id"] = gpc_df.iloc[topk_idx[i,j]][class_id_col]
            base[f"class_{j+1}_name"] = gpc_df.iloc[topk_idx[i,j]][class_name_col]
            base[f"score_{j+1}"] = float(topk_scores[i,j])
        rows.append(base)
    return pd.DataFrame(rows)

# ------------------------
# 3. Example usage
# ------------------------
if __name__ == "__main__":
    # Example: load your data
    # Replace this with a SELECT from ClearScape if needed
    
    tdf = DataFrame.from_table("products", schema_name="demo_user")
    df = tdf.to_pandas()

    gpc_df = pd.DataFrame({
        "class_id": [10000101, 10000234, 10000345],
        "class_name": ["Shampoos & Conditioners", "Laundry Detergents", "Headphones"],
        "class_text": [
            "shampoos and conditioners hair washing hair care products",
            "laundry detergents powder liquid laundry cleaning clothes",
            "headphones earphones wireless bluetooth personal audio devices"
        ]
    })

    pred_df = icf_tdcb_predict(gpc_df, df, k=3)
    print(pred_df.head())

    # Save results if you want
    pred_df.to_csv("icf_tdcb_predictions.csv", index=False)