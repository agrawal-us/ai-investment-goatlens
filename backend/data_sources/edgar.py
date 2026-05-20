"""
SEC EDGAR filing client.

Fetches 10-K and 10-Q filings via edgartools, chunks them into ~500-word
segments, embeds them with OpenAI text-embedding-3-small, and stores them
in Postgres via db.postgres. Repeat calls for the same ticker are no-ops
(permanent cache via upsert_chunk existence check).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai

from db.postgres import upsert_chunk

_executor = ThreadPoolExecutor(max_workers=2)

CHUNK_WORDS = 500
OVERLAP_WORDS = 50


class EDGARClient:
    def __init__(self) -> None:
        self._openai = openai.AsyncOpenAI()

    async def _run_sync(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, func, *args)

    async def _embed(self, text: str) -> List[float]:
        resp = await self._openai.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding

    def _fetch_sections(self, ticker: str) -> Dict[str, Any]:
        """Sync: pull 10-K and 10-Q sections via edgartools."""
        from edgar import Company  # noqa: PLC0415  (imported in executor thread)

        company = Company(ticker)
        results: Dict[str, Any] = {}

        # Latest 10-K
        filing = company.get_filings(form="10-K").latest()
        if filing:
            period = filing.period_of_report[:4]  # "YYYY" from "YYYY-MM-DD"
            tenk = filing.obj()
            results["10-K"] = {
                "period": period,
                "sections": {
                    "mda": tenk.mda or "",
                    "risk_factors": tenk.risk_factors or "",
                },
            }

        # Latest 10-Q
        filing = company.get_filings(form="10-Q").latest()
        if filing:
            d = datetime.strptime(filing.period_of_report, "%Y-%m-%d")
            quarter = (d.month - 1) // 3 + 1
            period = f"Q{quarter} {d.year}"
            tenq = filing.obj()
            results["10-Q"] = {
                "period": period,
                "sections": {
                    "mda": tenq.mda or "",
                },
            }

        return results

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + CHUNK_WORDS])
            if chunk.strip():
                chunks.append(chunk)
            i += CHUNK_WORDS - OVERLAP_WORDS
        return chunks

    async def index_filings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch, chunk, embed, and store SEC filings for ticker.
        Returns metadata dict or None on any failure.
        Subsequent calls for the same ticker are cheap — upsert_chunk skips
        already-stored chunks.
        """
        try:
            sections_data = await self._run_sync(self._fetch_sections, ticker)
            total_chunks = 0
            metadata: Dict[str, Any] = {
                "latest_10k": None,
                "latest_10q": None,
                "total_chunks": 0,
            }

            for filing_type, data in sections_data.items():
                period = data["period"]
                if filing_type == "10-K":
                    metadata["latest_10k"] = period
                else:
                    metadata["latest_10q"] = period

                for content in data["sections"].values():
                    if not content:
                        continue
                    chunks = self._chunk_text(content)
                    for idx, chunk in enumerate(chunks):
                        embedding = await self._embed(chunk)
                        await upsert_chunk(
                            ticker=ticker,
                            filing_type=filing_type,
                            period=period,
                            chunk_index=total_chunks + idx,
                            content=chunk,
                            embedding=embedding,
                        )
                    total_chunks += len(chunks)

            metadata["total_chunks"] = total_chunks
            return metadata
        except Exception:
            return None
