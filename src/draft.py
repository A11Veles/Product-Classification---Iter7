import pandas as pd
from teradataml import *
import ast
import json
import torch
from teradataml.dataframe.copy_to import copy_to_sql
from sklearn.metrics import f1_score

from modules.db import TeradataDatabase
from modules.models import( 
    OpusTranslationModelConfig, 
                        OpusTranslationModel, 
                        SentenceEmbeddingConfig, 
                        SentenceEmbeddingModel, 
)
from utils import clean_text, load_embedding_model, unicode_clean, load_translation_model
from constants import (
    CLEANED_TEST_DATA_PATH, TRAIN_VAL_DATA_PATH, CLASS_EMBEDDINGS_PATH, PRODUCT_TEST_EMBEDDINGS_PATH, 
    CLEANED_GPC_PATH, CLEANED_TEST_DATA_PATH, TEST_DATA_PATH, E5_LARGE_INSTRUCT_CONFIG_PATH, 
    OPUS_TRANSLATION_CONFIG_PATH, DATA_PATH, PRODUCT_TRAIN_EMBEDDINGS_PATH, QWEN3_8B_CONFIG_PATH, VALIDATION_DATA_PATH,
    FULL_DATA_SET_DATA_PATH, PRODUCT_FULL_DATASET_EMBEDDINGS_PATH,  CLEANED_FULL_DATASET_DATA_PATH,
    CLASS_EMBEDDINGS_PATH_QWEN, PRODUCT_FULL_DATASET_EMBEDDINGS__QWEN_PATH, TD_DB

)


td_db = TeradataDatabase() 
td_db.connect()
    

def insert_products_in_db():
    df = pd.read_csv(TRAIN_VAL_DATA_PATH)
    df.rename(columns={"Item_Name": "product_name"}, inplace=True)

    df.drop_duplicates(subset=["product_name"], inplace=True)
    df.dropna(subset=["product_name"], inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    df = df[["id" , "product_name"]]

    df['product_name'] = df['product_name'].apply(unicode_clean)

    copy_to_sql(df, "products", "demo_user", if_exists="replace")

def clean_products_in_db(): 

    tdf = DataFrame.from_table("products", schema_name=TD_DB)

    cleaning_query = """
    UPDATE demo_user.products
    SET product_name =
                    TRIM(
                        REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(product_name, '[[:digit:]]+', ''), 
                            '[-_/\\|]', ''),                              
                        '[[:punct:]]', ''                              
                        )
                    )
                    ;
    """

    td_db.execute_query(cleaning_query)

    tdf = tdf.assign(
    product_name = tdf.product_name.str.lower()
    )

    tdf_stripped = tdf.assign(
    product_name = tdf.product_name.str.strip()
    )        

    copy_to_sql(tdf, "products", "demo_user", if_exists="replace")


def translate_products():
    tdf = DataFrame.from_table("products", schema_name=TD_DB)
    df = tdf.to_pandas()

    products = df["product_name"].tolist()

    model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)

    batch_size = 32
    translations = []
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        batch_translations = [model.translate(p) for p in batch]
        translations.extend(batch_translations)

    df["translated_name"] = translations

    df = df[["id", "translated_name"]]

    copy_to_sql(df, "products", "demo_user", if_exists="replace")

    
def insert_classes_in_db():
    # Inserting the labels
    df = pd.read_csv(TRAIN_VAL_DATA_PATH)
    df_class = df["class"].dropna().unique()
    df_classes = pd.DataFrame({"class_name": df_class})
    df_classes["id"] = df_classes.index
    df_classes = df_classes[["id", "class_name"]]

    copy_to_sql(df_classes, "classes", "demo_user", if_exists="replace")

    # Inserting the truth value for each row
    df_actual_class = df[["class"]]
    df_actual_class.rename(columns={'class': 'class_name'}, inplace=True)
    df_actual_class["product_id"] = df_actual_class.index
    df_actual_class = df_actual_class[["product_id", "class_name"]]

    copy_to_sql(df_actual_class, "actual_classes", "demo_user", if_exists="replace")

def clean_classes_in_db(): 

    tdf = DataFrame.from_table("classes", schema_name=TD_DB)

    cleaning_query = """
    UPDATE demo_user.classes
    SET class_name =
                    TRIM(
                        REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(class_name, '[[:digit:]]+', ''), 
                            '[-_/\\|]', ''),                              
                        '[[:punct:]]', ''                              
                        )
                    )
                    ;
    """

    td_db.execute_query(cleaning_query)

    tdf = tdf.assign(
    class_name = tdf.class_name.str.lower()
    )

    tdf_stripped = tdf.assign(
    class_name = tdf.class_name.str.strip()
    )        

    copy_to_sql(tdf, "classes", "demo_user", if_exists="replace")


    tdf = DataFrame.from_table("actual_classes", schema_name=TD_DB)

    cleaning_query = """
    UPDATE demo_user.actual_classes
    SET class_name =
                    TRIM(
                        REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(class_name, '[[:digit:]]+', ''), 
                            '[-_/\\|]', ''),                              
                        '[[:punct:]]', ''                              
                        )
                    )
                    ;
    """

    td_db.execute_query(cleaning_query)

    tdf = tdf.assign(
    class_name = tdf.class_name.str.lower()
    )

    tdf_stripped = tdf.assign(
    class_name = tdf.class_name.str.strip()
    )        

    copy_to_sql(tdf, "actual_classes", "demo_user", if_exists="replace")


def create_product_embeddings():

    tdf = DataFrame.from_table("products", schema_name=TD_DB)
    df = tdf.to_pandas()

    model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)

    products = df["translated_name"].tolist()
    embeddings = model.get_embeddings(products)

    df["embeddings"] = embeddings.tolist()

    df.to_csv(PRODUCT_FULL_DATASET_EMBEDDINGS_PATH)

def load_product_emebddings():
    df = pd.read_csv(PRODUCT_FULL_DATASET_EMBEDDINGS_PATH)

    df["embeddings"] = df["embeddings"].apply(ast.literal_eval)

    emb_cols = pd.DataFrame(df["embeddings"].to_list(), columns=[f"embed_{i}" for i in range(len(df["embeddings"][0]))])

    df_expanded = pd.concat([df[['id']], emb_cols], axis=1)

    copy_to_sql(df_expanded, "p_embeddings", "demo_user", if_exists="replace")

def load_class_emebddings():
    df = pd.read_csv(TRAIN_VAL_DATA_PATH)

    df_class = df["class"].dropna().unique()
    df = pd.DataFrame({"class": df_class})

    model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)

    classes = df["class"].tolist()
    embeddings = model.get_embeddings(classes)

    embeddings = embeddings.tolist()
    emb_cols = pd.DataFrame(embeddings, columns=[f'embed_{i}' for i in range(len(embeddings[0]))])

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    df_expanded = pd.concat([df[['id']], emb_cols], axis=1)

    copy_to_sql(df_expanded, "c_embeddings", "demo_user", if_exists="replace")

def calculate_in_db_similiraty():
    result_table = "demo_user.similiratiy_score"
    q = f"""
    CREATE TABLE {result_table} (
        item_id BIGINT,
        closest_category_id BIGINT,
        cosine_distance FLOAT
    );
    """
    td_db.execute_query(q)
    print("***Similarty table created")
    vector_cols = ", ".join([f"embed_{i}" for i in range(1024)])

    vector_cols_quoted = ", ".join([f"'embed_{i}'" for i in range(1024)])

    classification_sql = f"""
    INSERT INTO {result_table}
    WITH RankedDistances AS (
        SELECT
            o.Target_ID AS product_id,
            o.Reference_ID AS class_id,
            o.Distance,
            ROW_NUMBER() OVER (PARTITION BY o.Target_ID ORDER BY o.Distance ASC) as rn
        FROM TD_VectorDistance (
            ON (SELECT id, {vector_cols} FROM p_embeddings) AS TargetTable
            ON (SELECT id, {vector_cols} FROM c_embeddings) AS ReferenceTable DIMENSION
            USING
                TargetIDColumn('id')
                RefIDColumn('id')
                TargetFeatureColumns({vector_cols_quoted})
                RefFeatureColumns({vector_cols_quoted})
                DistanceMeasure('cosine')
        ) AS o
    )
    SELECT
        product_id,
        class_id,
        Distance
    FROM RankedDistances
    WHERE rn = 1;
    """
    td_db.execute_query(classification_sql)

def results():
    results_query = f"""
    SELECT
        p.translated_name AS product_name,
        c.class_name AS predicted_class,
        a.class_name AS actual_class,
        r.cosine_distance AS similarity_score
    FROM similiratiy_score r
    JOIN products p
        ON r.item_id = p.id
    JOIN classes c
        ON r.closest_category_id = c.id
    JOIN actual_classes a
        ON a.product_id = p.id;
    """

    tdf = td_db.execute_query(results_query)
    df = pd.DataFrame(tdf)
    df.dropna(inplace= True)

    return df

def calulcate_f1(df):
    y_pred = df["predicted_class"].tolist()
    y_true = df["actual_class"].tolist()

    return f1_score(y_true, y_pred, average="weighted")

    

def disconnect_from_db():
    td_db.disconnect()


def insert_to_db():
    insert_products_in_db()
    clean_products_in_db()
    insert_classes_in_db()
    clean_classes_in_db()

def translation_and_embeddings():
    create_product_embeddings()
    load_product_emebddings()
    load_class_emebddings()

def prepare_data():
    print("Preparing data: inserting...")
    insert_products_in_db()
    insert_classes_in_db()
    print("Preparing data: inserting Done, cleaning...")
    clean_products_in_db()
    clean_classes_in_db()
    print("Preparing data: cleaning Done, translating...")
    translate_products()
    print("Preparing data: translating Done, and generating embeddings...")
    create_product_embeddings()
    print("Preparing data: product embeddings Done, and loading embeddings...")
    load_product_emebddings()
    print("Preparing data: loading product embeddings Done, generating class embeddings...")
    load_class_emebddings()
    print("Data preparation complete.")

def run_analysis():
    print("Running analysis: calculating similarity and F1 score...")
    calculate_in_db_similiraty()
    print("Cosine Similarity Done, generating F1 score...")
    df = results()
    f1_score = calulcate_f1(df)
    print(f"F1 Score: {f1_score}")
    print(df.head(20))
    print("Analysis complete.")


def main():
    prepare_data()
    run_analysis()
    # print("Starting pipeline...")
    # translate_products()
    # insert_classes_in_db()
    # clean_classes_in_db()
    # create_product_embeddings()
    # print("Here...")
    # load_product_emebddings()
    # print("loading in DB...")
    # load_class_emebddings()
    # calculate_in_db_similiraty()
    # df = results()
    # f1_score = calulcate_f1(df)
    # print(f"F1 Score: {f1_score}")
    # print(df.head(20))   

main()

