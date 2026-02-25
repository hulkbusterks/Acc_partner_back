from sqlmodel import SQLModel, create_engine, Session
from contextlib import contextmanager
from typing import Generator
import os
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./acc_poc.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        pool_timeout=30,
    )


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a DB session that is always properly closed."""
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
