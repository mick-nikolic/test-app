from model.pipeline.preparation import prepare_df
from config.config import settings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
import pickle


def build_model() -> BaseEstimator:
    X_train, X_test, y_train, y_test = prepare_df()
    model = LogisticRegression(max_iter=2_000, random_state=12).fit(X_train, y_train)
    return model


def save_model(model: BaseEstimator) -> None:
    pickle.dump(model, open(f"{settings.model_path}/{settings.model_name}", "wb"))


# Main guard
if __name__ == "__main__":
    model = build_model()

    X_train, X_test, y_train, y_test = prepare_df()
    yhat = model.predict(X_test)
    print(classification_report(y_test, yhat))

    save_model(model)
