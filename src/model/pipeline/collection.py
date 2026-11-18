import pandas as pd
from sqlalchemy import select
from loguru import logger

from config.config import engine
from db.db_model import Iris


def load_dataset() -> pd.DataFrame:
    logger.info("Loading the Iris database.")
    query = select(Iris)
    return pd.read_sql(query, engine)


# Main guard
if __name__ == "__main__":
    df = load_dataset()
    print(df.head(2))
    print(f"The type of the data file: {type(df)};")
