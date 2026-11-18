from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath
from loguru import logger
from sqlalchemy import create_engine


# Parameterization class
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env", env_file_encoding="utf-8"
    )

    model_path: DirectoryPath
    model_name: str
    seed: int
    train_size: float
    database: str
    table: str


settings = Settings()

# Engine object (connection to SQLite)
engine = create_engine(settings.database)  # an absolute path to db
## allows the app to locate and execute the db file accuratelly

logger.remove()
logger.add(
    "app.log", rotation="1 day", retention="1 week", compression="zip", level="DEBUG"
)
