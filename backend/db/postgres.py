import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://postgres@localhost:5432/goatlens"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class FilingChunk(Base):
    __tablename__ = "filings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker = Column(String, nullable=False)
    filing_type = Column(String, nullable=False)  # "10-K" or "10-Q"
    period = Column(String, nullable=False)        # e.g. "2024", "Q3 2024"
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # Vector column created via raw DDL below — pgvector type not declared here
    # to avoid import-time dependency on pgvector package being installed
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    telegram_user_id = Column(String, primary_key=True)
    openai_api_key = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # Add embedding column with pgvector type if it doesn't exist
        await conn.execute(text(
            "ALTER TABLE filings ADD COLUMN IF NOT EXISTS "
            "embedding vector(1536)"
        ))


@asynccontextmanager
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def upsert_chunk(
    ticker: str,
    filing_type: str,
    period: str,
    chunk_index: int,
    content: str,
    embedding: List[float],
) -> None:
    async with get_db() as db:
        result = await db.execute(
            text(
                "SELECT id FROM filings "
                "WHERE ticker=:ticker AND filing_type=:ft "
                "AND period=:period AND chunk_index=:ci"
            ),
            {"ticker": ticker, "ft": filing_type, "period": period, "ci": chunk_index},
        )
        if result.fetchone():
            return
        await db.execute(
            text(
                "INSERT INTO filings (id, ticker, filing_type, period, chunk_index, content, embedding) "
                "VALUES (:id, :ticker, :ft, :period, :ci, :content, CAST(:emb AS vector))"
            ),
            {
                "id": str(uuid.uuid4()),
                "ticker": ticker,
                "ft": filing_type,
                "period": period,
                "ci": chunk_index,
                "content": content,
                "emb": str(embedding),
            },
        )


async def search_similar(
    query_embedding: List[float], ticker: str, top_k: int = 3
) -> List[str]:
    async with get_db() as db:
        result = await db.execute(
            text(
                "SELECT content FROM filings "
                "WHERE ticker = :ticker "
                "ORDER BY embedding <=> CAST(:emb AS vector) "
                "LIMIT :k"
            ),
            {"ticker": ticker, "emb": str(query_embedding), "k": top_k},
        )
        rows = result.fetchall()
    return [row[0] for row in rows]
