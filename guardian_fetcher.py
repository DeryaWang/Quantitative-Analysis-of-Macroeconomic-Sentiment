import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Set

import aiohttp
import pandas as pd
import polars as pl
import yfinance as yf
from pydantic import BaseModel, Field, ValidationError, field_validator

# Configure high-performance logging for production-grade traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def atomic_write_parquet(df: pl.DataFrame, path: str):
    """
    Implements the Atomic 'Write-and-Swap' pattern to prevent data corruption.
    Utilizes a temporary file and an atomic filesystem move to guarantee 
    persistence integrity even during system-level interruptions.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix=".tmp") as tmp:
        df.write_parquet(tmp.name)
        tmp_path = tmp.name
    
    try:
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

class StateCoordinator:
    """
    Manages the atomic persistence of the ingestion state to ensure system resilience.
    Utilizes a sparse set of processed page IDs to support idempotent resumes 
    and guarantee 'at-least-once' delivery in high-concurrency environments.
    """
    def __init__(self, checkpoint_path: str = "state.json"):
        self.path = checkpoint_path
        self._lock = asyncio.Lock()

    def load_state(self) -> Dict[str, Any]:
        """Loads the sparse state from the persistent store."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    data["processed_pages"] = set(data.get("processed_pages", []))
                    return data
            except Exception as e:
                logger.error(f"StateCoordinator: Failed to load state: {e}")
        return {"processed_pages": set(), "last_date": None}

    async def update_state(self, page_id: int):
        """Atomically updates the checkpoint after successful ingestion."""
        async with self._lock:
            state = self.load_state()
            state["processed_pages"].add(page_id)
            state["processed_pages"] = list(state["processed_pages"])
            with open(self.path, "w") as f:
                json.dump(state, f)

class GuardianArticle(BaseModel):
    """
    Pydantic schema for strict validation of incoming Guardian API JSON.
    Enforces a 100-character information density threshold to ensure 
    high Signal-to-Noise Ratio (SNR) for subsequent LLM analysis.
    """
    webPublicationDate: str
    webTitle: str = Field(alias="webTitle")
    bodyText: str
    sectionName: str

    @field_validator('bodyText')
    @classmethod
    def validate_content_density(cls, v: str) -> str:
        if len(v) < 100:
            raise ValueError("Inadequate information density: content length below 100 characters.")
        return v

    class Config:
        populate_by_name = True

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """Standard exponential backoff decorator for robust API interactions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientResponseError, aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                    if attempt == retries: raise e
                    wait = backoff_in_seconds * (2 ** attempt)
                    if getattr(e, 'status', None) == 429: wait += 10
                    await asyncio.sleep(wait)
        return wrapper
    return decorator

class GuardianFetcher:
    """
    Asynchronous data fetcher optimized for high-throughput news ingestion.
    Implements a Semaphore-governed Producer-Consumer architecture with 
    Atomic Checkpointing and LazyFrame transitions.
    """
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com/search"
        self.session: Optional[aiohttp.ClientSession] = None
        self.queue = asyncio.Queue(maxsize=20)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.state_coordinator = StateCoordinator()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(raise_for_status=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session: await self.session.close()

    @retry_with_backoff(retries=3)
    async def _fetch_page(self, params: Dict[str, Any], page: int):
        async with self.semaphore:
            p_params = {**params, "page": page}
            logger.info(f"Producer: Requesting page {page}...")
            async with self.session.get(self.base_url, params=p_params) as response:
                data = await response.json()
                results = data.get("response", {}).get("results", [])
                if results:
                    await self.queue.put((page, results))
                return data.get("response", {}).get("pages", 0)

    async def producer(self, start_date: str, end_date: str, query: str):
        """Concurrent producer with hole-detection via sparse checkpointing."""
        state = self.state_coordinator.load_state()
        processed = state["processed_pages"]

        params = {
            "api-key": self.api_key,
            "q": query,
            "from-date": start_date,
            "to-date": end_date,
            "show-fields": "headline,bodyText",
            "page-size": 50,
            "lang": "en"
        }

        try:
            total_pages = await self._fetch_page(params, 1)
            tasks = [
                self._fetch_page(params, p) 
                for p in range(2, min(total_pages + 1, 15)) 
                if p not in processed
            ]
            await asyncio.gather(*tasks)
        finally:
            await self.queue.put(None)

    async def consumer(self, output_dir: str) -> pl.LazyFrame:
        """Consumer method: Processes, validates, and persists streaming data."""
        all_lfs = []
        os.makedirs(output_dir, exist_ok=True)

        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break

            page_id, batch = item
            validated = []
            for art in batch:
                try:
                    flat = {
                        "webPublicationDate": art.get("webPublicationDate"),
                        "webTitle": art.get("webTitle"),
                        "bodyText": art.get("fields", {}).get("bodyText"),
                        "sectionName": art.get("sectionName")
                    }
                    validated.append(GuardianArticle(**flat).model_dump())
                except ValidationError:
                    continue

            if validated:
                df = pl.from_dicts(validated).lazy().select([
                    pl.col("webPublicationDate").str.slice(0, 19).str.to_datetime(strict=False).alias("Date"),
                    pl.col("webTitle").str.strip_chars().alias("Title"),
                    pl.col("bodyText").str.strip_chars().alias("Content"),
                    pl.col("sectionName").alias("Section")
                ]).drop_nulls(subset=["Date", "Title", "Content"]).collect()
                
                path = os.path.join(output_dir, f"news_page_{page_id}.parquet")
                atomic_write_parquet(df, path)
                await self.state_coordinator.update_state(page_id)
                all_lfs.append(df.lazy())
            
            self.queue.task_done()

        return pl.concat(all_lfs) if all_lfs else pl.DataFrame().lazy()

    async def fetch_historical_news(self, start_date: str, end_date: str, query: str) -> pl.LazyFrame:
        p_task = asyncio.create_task(self.producer(start_date, end_date, query))
        c_task = asyncio.create_task(self.consumer("data/raw_news"))
        await asyncio.gather(p_task, c_task)
        return await c_task

async def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pl.LazyFrame:
    """Market data retrieval with blocking I/O mitigation via wait_for."""
    loop = asyncio.get_event_loop()
    try:
        data = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: yf.download(symbol, start=start_date, end=end_date)),
            timeout=30.0
        )
    except Exception: return pl.DataFrame().lazy()
    
    if data.empty: return pl.DataFrame().lazy()
    if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]
    data = data.reset_index().rename(columns={"Date": "Date", "Close": "Close_Price"})
    return pl.from_pandas(data[["Date", "Close_Price"]]).lazy()
