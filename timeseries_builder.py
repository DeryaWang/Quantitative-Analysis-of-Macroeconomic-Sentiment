import logging
from typing import Any, Dict, List, Optional
import math
import polars as pl

# Configure high-performance logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesBuilder:
    """
    Advanced Quantitative Time-Series Engine.
    Utilizes Polars' Pure Lazy API to enforce causal integrity and 
    implements physics-inspired signal dynamics for macroeconomic research.
    """

    @staticmethod
    def aggregate_daily_sentiment(lf: pl.LazyFrame, half_life_days: int = 3) -> pl.LazyFrame:
        """
        Calculates daily sentiment using SNR Optimization (Confidence-Weighted Averaging).
        Implements Information Decay via EMA with alpha derived from a 3-day Half-life.
        """
        alpha = 1 - math.exp(math.log(0.5) / half_life_days)
        
        return (
            lf
            .with_columns(pl.col("Date").cast(pl.Date))
            .with_columns((pl.col("score") * pl.col("confidence")).alias("weighted_score"))
            .group_by("Date")
            .agg([
                (pl.col("weighted_score").sum() / pl.col("confidence").sum()).alias("daily_sentiment"),
                pl.col("score").count().alias("article_count")
            ])
            .with_columns(pl.col("daily_sentiment").fill_nan(0.0).fill_null(0.0))
            .sort("Date")
            # Information Decay: Contextual EMA based on narrative persistence
            .with_columns(
                pl.col("daily_sentiment")
                .ewm_mean(alpha=alpha, ignore_nulls=True)
                .alias("smoothed_recession_fear")
            )
            # Signal Velocity: First derivative of the smoothed index
            .with_columns(
                pl.col("smoothed_recession_fear").diff().alias("sentiment_velocity")
            )
            # Causal Rolling Normalization: 20-day Rolling Z-Score
            .with_columns([
                ((pl.col("smoothed_recession_fear") - 
                  pl.col("smoothed_recession_fear").rolling_mean(window_size=7)) / 
                 pl.col("smoothed_recession_fear").rolling_std(window_size=7))
                .alias("sentiment_zscore")
            ]))

    @staticmethod
    def align_market_to_sentiment(sentiment_lf: pl.LazyFrame, market_lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Point-in-Time Data Aligner:
        Synchronizes sentiment signals with market state using pl.join_asof (Backward).
        Ensures strict causality and aligns Date schemas.
        """
        return (
            sentiment_lf.with_columns(pl.col("Date").cast(pl.Date)).sort("Date")
            .join_asof(
                market_lf.with_columns(pl.col("Date").cast(pl.Date)).sort("Date"),
                on="Date",
                strategy="backward"
            )
            .with_columns(pl.col("Close_Price").fill_null(strategy="forward"))
            .sort("Date")
            # 1. Logarithmic Returns for time-additivity
            .with_columns(
                (pl.col("Close_Price") / pl.col("Close_Price").shift(1)).log().alias("log_return")
            )
            # 2. Volatility Scaling: Normalizing exposure by 7-day annualized vol
            .with_columns(
                (pl.col("log_return").rolling_std(window_size=7) * math.sqrt(252)).alias("annual_vol")
            )
            # 3. Risk-Scaled Signal
            .with_columns(
                (pl.col("sentiment_zscore") / pl.col("annual_vol")).alias("risk_scaled_signal")
            )
        )

    @staticmethod
    def calculate_metrics(lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Institutional Performance Attribution:
        Calculates Annualized Sharpe Ratio and Maximum Drawdown (MDD).
        """
        return (
            lf
            .with_columns(
                (pl.col("log_return") * pl.col("sentiment_zscore").shift(1)).alias("strategy_return")
            )
            .with_columns(
                pl.col("strategy_return").cum_sum().alias("cum_strategy_return")
            )
            .with_columns(
                (pl.col("cum_strategy_return").exp()).alias("equity_curve")
            )
            .with_columns(
                (pl.col("equity_curve") / pl.col("equity_curve").cum_max() - 1).alias("drawdown")
            )
        )

    @staticmethod
    def get_summary(lf: pl.LazyFrame) -> Dict[str, float]:
        """Final execution and metric extraction."""
        res = lf.select([
            (pl.col("strategy_return").mean() / pl.col("strategy_return").std() * math.sqrt(252)).alias("sharpe"),
            pl.col("drawdown").min().alias("max_drawdown")
        ]).collect()
        
        return res.to_dicts()[0]
