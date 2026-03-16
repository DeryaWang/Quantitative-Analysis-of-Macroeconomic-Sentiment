# Quantitative Analysis of Macroeconomic Sentiment: An LLM-Driven Approach

## Overview
This repository implements a high-throughput, asynchronous pipeline that quantifies macroeconomic narratives (specifically "Recession Fears") to evaluate their predictive capacity against US equity market returns. It bridges unstructured Natural Language Processing (NLP) with systematic quantitative trading using physics-inspired signal dynamics.

## System Architecture
The pipeline relies on asynchronous I/O and a "Lazy Execution" graph for memory-efficient, out-of-core processing:

* **guardian_fetcher.py:** Handles concurrent, rate-limited API consumption using `aiohttp` and `asyncio`, utilizing an atomic write-and-swap pattern for data integrity.
* **llm_analyzer.py:** Executes the LLM inference engine, enforcing deterministic, JSON-structured sentiment extraction via strict `pydantic` schemas.
* **timeseries_builder.py:** Constructs the quantitative time-series using `polars` LazyFrames. Implements confidence-weighted averaging (SNR optimization), a 3-day half-life Exponential Moving Average (EMA), and enforces strict Point-in-Time (PIT) integrity via `join_asof(strategy="backward")` to eliminate look-ahead bias.

## Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txtConfigure Environment Variables:
Ensure the following API keys are accessible in your environment:

GUARDIAN_API_KEY

OPENAI_API_KEY

Execution:
Execute the primary Jupyter Notebook (pre.ipynb) sequentially to ingest data, run inference, compile the time-series, and generate performance attribution and divergence analytics.

Computational Complexity
Designed for massive scale, the system's temporal alignment and PIT joins operate in O(MlogM+NlogN+M+N) time. It is strictly bounded by the Apache Arrow columnar memory footprint, ensuring O(1) copy overhead during execution graph materialization.
