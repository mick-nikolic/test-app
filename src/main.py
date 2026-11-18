from model import model_service
from model.pipeline import preparation

from loguru import logger
from sklearn.metrics import fbeta_score

if __name__ == "__main__":
    model = model_service.ModelService()
    model.load_model()

    print(type(model.model))

    X_tr, X_te, y_tr, y_te = preparation.prepare_df()

    try:
        yhat = model.predict(X_te)
    except Exception as e:
        logger.error(f"{e}.")

    print(round(fbeta_score(y_te, yhat, beta=1, average="macro"), 2))
