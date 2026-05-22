import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Any, List

import openai

from db.postgres import upsert_chunk

_executor = ThreadPoolExecutor(max_workers=2)
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50


class EDGARClient:
    def __init__(self):
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
        """Sync: use edgartools to pull 10-K and 10-Q sections."""
        print(f"[EDGAR] _fetch_sections: importing edgar and creating Company({ticker})")
        import edgar as edgar_module
        identity = os.getenv("EDGAR_IDENTITY", "GOATlens contact@goatlens.dev")
        edgar_module.set_identity(identity)
        print(f"[EDGAR] _fetch_sections: identity set to '{identity}'")
        from edgar import Company
        company = Company(ticker)
        results = {}
        print(f"[EDGAR] _fetch_sections: fetching 10-K filings for {ticker}")

        filing = company.get_filings(form="10-K").latest()
        if filing:
            period = filing.period_of_report[:4]  # "YYYY" from "YYYY-MM-DD"
            tenk = filing.obj()
            results["10-K"] = {
                "period": period,
                "sections": {
                    "mda": tenk.management_discussion or "",
                    "risk_factors": tenk.risk_factors or "",
                },
            }
            print(f"[EDGAR] 10-K fetched: period={period}, mda_len={len(tenk.management_discussion or '')}, rf_len={len(tenk.risk_factors or '')}")

        filing = company.get_filings(form="10-Q").latest()
        if filing:
            d = datetime.strptime(filing.period_of_report, "%Y-%m-%d")
            quarter = (d.month - 1) // 3 + 1
            period = f"Q{quarter} {d.year}"
            tenq = filing.obj()
            # TenQ MD&A is Part I, Item 2 accessed via doc
            try:
                mda_section = tenq.doc.get_section("part_i_item_2")
                mda_text = mda_section.text() if mda_section else ""
            except Exception:
                mda_text = ""
            results["10-Q"] = {
                "period": period,
                "sections": {
                    "mda": mda_text,
                },
            }
            print(f"[EDGAR] 10-Q fetched: period={period}, mda_len={len(mda_text)}")

        return results

    def _chunk_text(self, text: str) -> List[str]:
        """Split into ~500-word chunks with 50-word overlap."""
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i : i + CHUNK_TOKENS])
            if chunk.strip():
                chunks.append(chunk)
            i += CHUNK_TOKENS - OVERLAP_TOKENS
        return chunks

    async def index_filings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch, chunk, embed, store. Returns metadata dict or None on failure."""
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
        except Exception as e:
            import traceback
            print(f"[EDGAR] index_filings failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
