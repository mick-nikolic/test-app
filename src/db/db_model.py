from sqlalchemy import REAL, INTEGER
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.config import settings


class Base(DeclarativeBase):
    pass


class Iris(Base):
    __tablename__ = settings.table

    sepal_length_cm: Mapped[float] = mapped_column(REAL(), primary_key=True)
    sepal_width_cm: Mapped[float] = mapped_column(REAL())
    petal_length_cm: Mapped[float] = mapped_column(REAL())
    petal_width_cm: Mapped[float] = mapped_column(REAL())
    target: Mapped[int] = mapped_column(INTEGER())
