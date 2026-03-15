import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import polars as pl
from guardian_fetcher import atomic_write_parquet, retry_with_backoff
from pydantic import BaseModel, Field, ValidationError

# Configure logging for production-grade traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentResult(BaseModel):
    """
    Pydantic schema for strict validation of LLM sentiment output.
    Ensures structural integrity and type safety for downstream quantitative analysis.
    """
    score: int = Field(description="Recession fear score: 1 (Fear), 0 (Neutral), -1 (Safe)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in analysis")
    sectors: List[str] = Field(default_factory=list, description="Identified economic sectors")
    reasoning: str = Field(description="Brief justification for the assigned score")

class LLMAnalyzer:
    """
    Asynchronous LLM engine optimized for fluid concurrency and row-level data integrity.
    Utilizes ID-based tracking and relational joining to ensure that sentiment 
    signals are strictly aligned with their corresponding news records.
    """
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://api.openai.com/v1", 
        model: str = "gpt-3.5-turbo",
        max_concurrent: int = 5
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.session = aiohttp.ClientSession(headers=headers, raise_for_status=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_system_prompt(self) -> str:
        return (
            "You are a Senior Quantitative Macroeconomist. Analyze the provided news text "
            "and extract the market's perception of 'US Recession Fears'. "
            "You MUST return ONLY a valid JSON object. Do not include markdown formatting. "
            "Schema: {\"score\": <int>, \"confidence\": <float>, \"sectors\": [<list>], \"reasoning\": \"<str>\"}"
        )

    @retry_with_backoff(retries=5, backoff_in_seconds=2) 
    async def _call_llm(self, text: str) -> Dict[str, Any]:
        """
        Executes a single LLM API call with Pydantic validation and token telemetry.
        Defensively strips Markdown artifacts using regex before JSON deserialization.
        """
        if not self.session:
            raise RuntimeError("ClientSession not initialized.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Analyze this macroeconomic text: {text[:1000]}"} 
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        async with self.semaphore:
            await asyncio.sleep(random.uniform(0.1, 0.5))
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                result = await response.json()
                usage = result.get("usage", {})
                telemetry = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
                content = result['choices'][0]['message']['content']
                sanitized_content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE)
                parsed_json = json.loads(sanitized_content)
                sentiment = SentimentResult(**parsed_json).model_dump()
                return {**sentiment, **telemetry}

    async def analyze_text(self, record: Dict[str, Any], row_id: int) -> Dict[str, Any]:
        """
        Wrapper for single text analysis with fallback defaults.
        Maintains row-level identity through the asynchronous lifecycle.
        """
        fallback = {
            "row_id": row_id, "score": 0, "confidence": 0.0, "sectors": [], 
            "reasoning": "Fallback.", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }
        try:
            result = await self._call_llm(record["Content"])
            return {"row_id": row_id, **result}
        except Exception as e:
            logger.error(f"Inference failed for row {row_id}: {e}")
            return fallback

    async def analyze_dataframe(
        self, 
        lf: pl.LazyFrame, 
        text_column: str, 
        output_dir: str = "data/processed_sentiment"
    ) -> pl.LazyFrame:
        """
        Orchestrates fluid batch inference over a Polars LazyFrame with ID-based alignment.
        Replaces horizontal concatenation with a Left Join to ensure strict row integrity.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Inject unique row_id to ensure relational integrity during asynchronous collection
        df = lf.collect().with_row_index(name="row_id")
        if df.is_empty():
            return df.lazy()

        rows = df.to_dicts()
        logger.info(f"Initiating ID-aligned inference stream for {len(rows)} records...")
        
        # Task Initialization with ID-based mapping
        tasks = [self.analyze_text(row, row["row_id"]) for row in rows]
        
        results = []
        batch_counter = 0
        checkpoint_size = 10
        
        # Stream Processing: Results are collected as they return from the API
        for completed_task in asyncio.as_completed(tasks):
            sentiment_data = await completed_task
            results.append(sentiment_data)
            batch_counter += 1
            
            # Atomic Persistence: Safeguarding intermediate data checkpoints
            if batch_counter % checkpoint_size == 0:
                current_results_df = pl.from_dicts(results[-checkpoint_size:])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(output_dir, f"batch_telemetry_{timestamp}.parquet")
                atomic_write_parquet(current_results_df, path)
        
        # Final Relational Alignment
        # Results are converted to a DataFrame and joined back on row_id to ensure order
        results_df = pl.from_dicts(results)
        
        return (
            df.lazy()
            .join(results_df.lazy(), on="row_id", how="left")
            .drop("row_id") # Clean schema for downstream processing
        )
