# run_ensemble.py (short)
import pandas as pd, numpy as np, torch
from teradataml import DataFrame
from src.modules.db import TeradataDatabase
from src.modules.models import SentenceEmbeddingModel, SentenceEmbeddingConfig, IcfTdcbVoter, IcfTdcbConfig, LLMVoter, LLMVoterConfig

def majority3(a, b, c):
    out = []
    for x, y, z in zip(a, b, c):
        out.append(x if (x==y or x==z) else (y if y==z else y))  # tie → embedding
    return np.array(out)

if __name__ == "__main__":
    TeradataDatabase().connect()

    # products
    df_prod = DataFrame.from_table("products", schema_name="demo_user").to_pandas()
    if "translated_name" not in df_prod.columns:
        raise ValueError("Add/rename your product text column to 'translated_name'.")

    # GPC (Class level)
    gpc = DataFrame.from_table("gpc_orig", schema_name="demo_user")[["ClassTitle","ClassDefinition","ClassCode"]].to_pandas()
    gpc["class_text"] = (gpc["ClassTitle"].fillna("") + ". " + gpc["ClassDefinition"].fillna("")).str.strip()
    class_titles = gpc["ClassTitle"].tolist()

    # 1) ICF·TDCB voter
    icf_idx, icf_score = IcfTdcbVoter(IcfTdcbConfig()).vote(gpc, df_prod)

    # 2) Embedding voter (your model)
    emb = SentenceEmbeddingModel(SentenceEmbeddingConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16" if torch.cuda.is_available() else "float32",
        model_id="intfloat/e5-base",
        truncate_dim=None,
        convert_to_numpy=True,
        convert_to_tensor=False,
        use_prompt=False
    ))
    class_emb = emb.get_embeddings(gpc["class_text"].tolist())
    prod_emb  = emb.get_embeddings(df_prod["translated_name"].tolist())
    class_emb = class_emb / (np.linalg.norm(class_emb, axis=1, keepdims=True)+1e-9)
    prod_emb  = prod_emb  / (np.linalg.norm(prod_emb,  axis=1, keepdims=True)+1e-9)
    S = prod_emb @ class_emb.T
    emb_idx, emb_score = S.argmax(1), S.max(1)

    # 3) LLM voter (free-tier via Groq; set GROQ_API_KEY)
    llm_idx = LLMVoter(LLMVoterConfig()).vote(df_prod, class_titles, product_text_col="translated_name")

    # Majority
    final_idx = majority3(icf_idx, emb_idx, llm_idx)

    out = pd.DataFrame({
        "product_text": df_prod["translated_name"],
        "icf_class":    [class_titles[i] for i in icf_idx],
        "emb_class":    [class_titles[i] for i in emb_idx],
        "llm_class":    [class_titles[i] for i in llm_idx],
        "final_class":  [class_titles[i] for i in final_idx],
    })
    out.to_csv("ensemble_predictions.csv", index=False)
    print("Saved ensemble_predictions.csv")