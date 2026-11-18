from model.pipeline.collection import load_dataset
from config.config import settings

from loguru import logger
from sklearn.model_selection import train_test_split


def prepare_df():

    # 1. Load dataset
    logger.info("Loading a dataset.")
    logger.debug("Breast cancer dataset.")
    df = load_dataset()

    # 2. Split train, test
    logger.info("Splitting predictors and target.")
    X = df.drop("target", axis=1)
    y = df.target
    logger.info("Train-test split.")
    logger.debug(
        f"Stratified split; Train size: {settings.train_size}; Seed for reproducibility: {settings.seed}"
    )
    logger.warning(
        "Note: proportion of the test set depends on the dataset size and your application; make sure to leave enough records for eval."
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, train_size=settings.train_size, random_state=settings.seed
    )

    # 3. Normalize with z-score noramlization
    logger.info("Input vector normalization.")
    X_tr_std = (X_tr - X_tr.mean()) / X_tr.std()
    X_te_std = (X_te - X_tr.mean()) / X_tr.std()

    return X_tr_std, X_te_std, y_tr, y_te


# Main guard
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = prepare_df()

    print(X_tr.head(2))
