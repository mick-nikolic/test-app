from pathlib import Path
import pickle
import numpy as np
from loguru import logger

from config.config import settings
from model.pipeline.model_def import build_model

# logger.add("info.log", rotation="500 B", retention="1 minute", compression="zip", level="INFO")
# logger.add("error.log", rotation="1 week", retention="1 month", compression="zip", level="ERROR")
# logger.add("critical.log", rotation="1 week", retention="1 month", compression="zip", level="CRITICAL")


class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self, model_name=settings.model_name) -> None:
        model_path = Path(settings.model_path) / model_name
        if model_path.exists():
            logger.info(f"Loading model from {model_path}.")
            self.model = pickle.load(open(model_path, "rb"))
        else:
            logger.warning(f"{model_path} not found; training a new model.")
            self.model = build_model()

    def predict(self, input_vector) -> np.ndarray:
        return self.model.predict(input_vector)
