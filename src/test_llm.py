from utils import load_llm_model
from constants import LLM_CONFIG_PATH
from modules.db import TeradataDatabase
import pandas as pd

model = load_llm_model(LLM_CONFIG_PATH)

td_db = TeradataDatabase()
td_db.connect()

print("Loading sample products...")
products_query = """
        SELECT TOP 10 id, translated_name 
        FROM demo_user.full_dataset_translated_products 
    """
products_tdf = td_db.execute_query(products_query)
products_df = pd.DataFrame(products_tdf)
print(f"Loaded {len(products_df)} products")

print("Loading GPC segments...")
gpc_query = """
SELECT SegmentTitle, MIN(SegmentDefinition) AS SegmentDefinition
FROM demo_user.gpc_orig
WHERE SegmentTitle IS NOT NULL
GROUP BY SegmentTitle;

    """
gpc_tdf = td_db.execute_query(gpc_query)
gpc_df = pd.DataFrame(gpc_tdf)
unique_segments = gpc_df['SegmentTitle'].unique().tolist()

df = model.predict(products_df, unique_segments, "translated_name")
print(gpc_df.head())