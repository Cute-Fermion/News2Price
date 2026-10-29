"""
特征工程模块
- 日度情感指数聚合
- 新闻强度（每日新闻条数）
- 指数衰减加权（处理非交易日新闻）
- 极端情绪标记
- 与行情数据对齐合并
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """从新闻情感数据构造日度特征"""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        feat_config = config.get("features", {})
        self.decay_lambda = feat_config.get("decay_lambda", 0.5)
        self.decay_window = feat_config.get("decay_window", 5)
        self.extreme_threshold = feat_config.get("extreme_threshold", 1.5)
        self.rolling_window = feat_config.get("rolling_window", 30)

    def aggregate_daily_sentiment(
        self,
        news_df: pd.DataFrame,
        score_col: str = "dict_score",
    ) -> pd.DataFrame:
        """
        将新闻情感分数按日聚合
        返回: date, avg_sentiment, sentiment_std, news_count, pos_ratio, neg_ratio
        """
        # 确保日期列存在
        if "date" not in news_df.columns:
            raise ValueError("news_df 必须包含 'date' 列")

        df = news_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 按日聚合
        daily = df.groupby("date").agg(
            avg_sentiment=(score_col, "mean"),
            sentiment_std=(score_col, "std"),
            news_count=(score_col, "count"),
            pos_count=(score_col, lambda x: (x > 0.1).sum()),
            neg_count=(score_col, lambda x: (x < -0.1).sum()),
        ).reset_index()

        # 计算正负面比例
        daily["pos_ratio"] = daily["pos_count"] / daily["news_count"]
        daily["neg_ratio"] = daily["neg_count"] / daily["news_count"]
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

        daily["date"] = pd.to_datetime(daily["date"])
        logger.info(f"日度情感聚合完成: {len(daily)} 个交易日有新闻")
        return daily

    def apply_decay(
        self,
        daily_sentiment: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
        score_col: str = "avg_sentiment",
    ) -> pd.Series:
        """
        对非交易日的情感进行指数衰减
        effective_sentiment_t = Σ sentiment_{t-k} × λ^k

        参数:
            daily_sentiment: 日度情感 DataFrame (date, avg_sentiment, ...)
            trading_dates: 完整交易日序列
            score_col: 情感得分列名
        """
        # 将情感数据设为 date 索引
        sentiment_series = daily_sentiment.set_index("date")[score_col]

        # 对每个交易日，计算衰减加权的累积情感
        result = []
        for date in trading_dates:
            weighted_sum = 0.0
            weight_total = 0.0
            for k in range(self.decay_window + 1):
                lookback_date = date - pd.Timedelta(days=k)
                # 向前查找最近的有新闻的日期
                while lookback_date >= date - pd.Timedelta(days=self.decay_window * 2):
                    if lookback_date in sentiment_series.index:
                        val = sentiment_series[lookback_date]
                        if pd.notna(val):
                            weight = self.decay_lambda ** k
                            weighted_sum += val * weight
                            weight_total += weight
                        break
                    lookback_date -= pd.Timedelta(days=1)

            if weight_total > 0:
                result.append(weighted_sum / weight_total)
            else:
                result.append(0.0)

        return pd.Series(result, index=trading_dates, name=f"{score_col}_decay")

    def create_extreme_flag(self, sentiment_series: pd.Series) -> pd.Series:
        """
        标记极端情绪日（使用滚动窗口，避免 look-ahead bias）
        基于过去 rolling_window 日的均值和标准差判断
        前 rolling_window-1 个数据点使用扩展窗口（expanding）
        """
        window = self.rolling_window

        # 滚动均值和标准差（min_periods=10 保证早期数据也能计算）
        rolling_mean = sentiment_series.rolling(window=window, min_periods=10).mean()
        rolling_std = sentiment_series.rolling(window=window, min_periods=10).std()

        # 对于前 window 个数据点，使用 expanding 窗口补充
        expanding_mean = sentiment_series.expanding(min_periods=5).mean()
        expanding_std = sentiment_series.expanding(min_periods=5).std()

        # 合并：前 window 个用 expanding，之后用 rolling
        mean = rolling_mean.combine_first(expanding_mean)
        std = rolling_std.combine_first(expanding_std)

        upper = mean + self.extreme_threshold * std
        lower = mean - self.extreme_threshold * std

        flag = pd.Series(0, index=sentiment_series.index, name="extreme_sentiment")
        flag[sentiment_series > upper] = 1   # 极端正面
        flag[sentiment_series < lower] = -1  # 极端负面
        return flag

    def build_features(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame,
        score_col: str = "dict_score",
    ) -> pd.DataFrame:
        """
        完整特征构建流程
        返回合并后的 DataFrame，包含:
        - 行情数据: trade_date, close, volume, pct_chg, return
        - 情感特征: avg_sentiment, sentiment_std, news_count, pos_ratio, neg_ratio
        - 衰减情感: avg_sentiment_decay
        - 极端情绪: extreme_sentiment
        - 哑变量: weekday_*, month_*
        """
        # 1. 日度情感聚合
        daily_sentiment = self.aggregate_daily_sentiment(news_df, score_col)

        # 2. 准备行情数据
        stock = stock_df.copy()
        stock["trade_date"] = pd.to_datetime(stock["trade_date"])
        stock = stock.sort_values("trade_date").reset_index(drop=True)

        # 计算收益率（如果还没有）
        if "return" not in stock.columns:
            stock["return"] = stock["pct_chg"] / 100  # 百分比转小数

        # 对数去趋势交易量（参照 Tetlock）
        stock["log_volume"] = np.log(stock["volume"] + 1)
        # 去趋势：减去滚动均值
        stock["volume_detrend"] = stock["log_volume"] - stock["log_volume"].rolling(20, min_periods=5).mean()
        stock["volume_detrend"] = stock["volume_detrend"].fillna(0)

        # 3. 合并情感数据到交易日
        merged = stock.merge(
            daily_sentiment,
            left_on="trade_date",
            right_on="date",
            how="left",
        )

        # 填充无新闻的日期（情感=0，新闻数=0）
        sentiment_cols = ["avg_sentiment", "sentiment_std", "news_count", "pos_ratio", "neg_ratio"]
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        # 4. 指数衰减情感
        trading_dates = stock["trade_date"]
        decay_series = self.apply_decay(daily_sentiment, trading_dates, "avg_sentiment")
        merged["avg_sentiment_decay"] = decay_series.values

        # 5. 极端情绪标记
        merged["extreme_sentiment"] = self.create_extreme_flag(merged["avg_sentiment"])
        merged["abs_extreme"] = merged["extreme_sentiment"].abs()  # 极端程度（用于交易量回归）

        # 6. 哑变量
        merged = self._add_dummies(merged)

        # 7. 清理
        merged = merged.drop(columns=["date"], errors="ignore")
        merged = merged.sort_values("trade_date").reset_index(drop=True)

        logger.info(f"特征构建完成: {len(merged)} 行, {len(merged.columns)} 列")
        return merged

    def _add_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加星期和月份哑变量"""
        df = df.copy()

        # 星期哑变量 (Mon=0, ..., Sun=6)
        weekday = df["trade_date"].dt.weekday
        for i in range(5):  # Mon-Fri
            df[f"weekday_{i}"] = (weekday == i).astype(int)

        # 月份哑变量
        month = df["trade_date"].dt.month
        for m in range(1, 13):
            df[f"month_{m}"] = (month == m).astype(int)

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 简单测试
    fe = FeatureEngineer()

    # 模拟数据
    dates = pd.date_range("2024-01-01", periods=10, freq="B")  # 工作日
    stock_df = pd.DataFrame({
        "trade_date": dates,
        "close": np.random.randn(10).cumsum() + 3000,
        "volume": np.random.randint(100000, 200000, 10),
        "pct_chg": np.random.randn(10),
    })

    news_dates = pd.date_range("2024-01-01", periods=20, freq="D")  # 包含周末
    news_df = pd.DataFrame({
        "date": news_dates,
        "dict_score": np.random.randn(20),
    })

    result = fe.build_features(stock_df, news_df)
    print(result[["trade_date", "return", "avg_sentiment", "news_count", "avg_sentiment_decay"]].head())
