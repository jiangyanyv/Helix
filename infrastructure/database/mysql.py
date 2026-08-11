"""MySQL 引擎工厂 + scoped session。

全局单例：所有 Memory Service 通过 get_db_session() 拿到同一个 scoped_session。
注意：SQLAlchemy 在 Windows Python 下使用 pymysql 驱动。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from config import Config


_engine: Engine | None = None
_session_factory: scoped_session[Session] | None = None


def create_mysql_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            Config.mysql_url(),
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=2,
            pool_recycle=1800,
            echo=False,
            future=True,
        )
    return _engine


def create_session_factory() -> scoped_session[Session]:
    global _session_factory
    if _session_factory is None:
        engine = create_mysql_engine()
        factory = sessionmaker(
            bind=engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        _session_factory = scoped_session(factory)
    return _session_factory


@contextmanager
def get_db_session() -> Iterator[Session]:
    """上下文管理器：with get_db_session() as sess: ..."""
    factory = create_session_factory()
    sess = factory()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
        factory.remove()
