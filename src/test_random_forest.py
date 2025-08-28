import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from modules.db import TeradataDatabase
from utils import load_tfidf_random_forest


def main():
    td_db = TeradataDatabase()
    td_db.connect()

    print("Loading sample products...")
    products_query = """
        SELECT id, translated_name 
        FROM demo_user.translated_products_test_dataset
    """
    products_tdf = td_db.execute_query(products_query)
    df = pd.DataFrame(products_tdf)
        
    X = df["text"]
    y = df["ClassTitle"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = load_tfidf_random_forest()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("F1 Score:", f1_score(y_test, preds, "weighted"))


if __name__ == "__main__":
    main()