"""
可视化模块
- 行情走势图
- 情感指数时序图
- 情感与收益散点图
- VAR 脉冲响应图
- 回归系数图
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.utils import load_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STSong", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """结果可视化"""

    def __init__(self, config: dict | None = None, output_dir: str | Path | None = None):
        if config is None:
            config = load_config()
        viz_config = config.get("visualization", {})
        self.dpi = viz_config.get("dpi", 150)
        self.format = viz_config.get("format", "png")

        if output_dir is None:
            output_dir = PROJECT_ROOT / "results" / "figures"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置风格
        style = viz_config.get("style", "seaborn-v0_8-whitegrid")
        try:
            sns.set_theme(style=style)
        except ValueError:
            sns.set_theme(style="whitegrid")

    def _save(self, fig, name: str):
        """保存图表"""
        path = self.output_dir / f"{name}.{self.format}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"保存图表: {path}")
        return path

    # ================================================================
    # 行情走势图
    # ================================================================
    def plot_stock_price(self, df: pd.DataFrame, title: str = "沪深300指数走势"):
        """绘制收盘价和交易量走势"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 1])

        ax1.plot(df["trade_date"], df["close"], linewidth=1, color="#1f77b4")
        ax1.set_ylabel("收盘价", fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 交易量
        colors = ["#2ca02c" if r >= 0 else "#d62728" for r in df.get("return", pd.Series(0))]
        ax2.bar(df["trade_date"], df["volume"] / 1e6, color=colors, alpha=0.6, width=1)
        ax2.set_ylabel("交易量 (百万)", fontsize=12)
        ax2.set_xlabel("日期", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 格式化 x 轴日期
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

        return self._save(fig, "stock_price")

    # ================================================================
    # 情感指数时序图
    # ================================================================
    def plot_sentiment_timeline(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "avg_sentiment",
        title: str = "日度情感指数",
    ):
        """绘制情感指数随时间变化"""
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(df["trade_date"], df[sentiment_col], linewidth=0.8, alpha=0.8, label=sentiment_col)

        # 零线
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # 极端情绪区域
        mean = df[sentiment_col].mean()
        std = df[sentiment_col].std()
        ax.axhline(y=mean + 1.5 * std, color="orange", linestyle=":", alpha=0.5, label="极端阈值")
        ax.axhline(y=mean - 1.5 * std, color="orange", linestyle=":", alpha=0.5)

        ax.set_ylabel("情感得分", fontsize=12)
        ax.set_xlabel("日期", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

        return self._save(fig, f"sentiment_{sentiment_col}")

    # ================================================================
    # 情感 vs 收益散点图
    # ================================================================
    def plot_sentiment_vs_return(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "avg_sentiment",
        lag: int = 1,
    ):
        """绘制滞后情感得分 vs 当日收益的散点图"""
        fig, ax = plt.subplots(figsize=(10, 7))

        sentiment_lag = df[sentiment_col].shift(lag)
        valid = df.dropna(subset=[sentiment_col, "return"])

        ax.scatter(
            sentiment_lag.loc[valid.index],
            valid["return"],
            alpha=0.3,
            s=20,
            color="#1f77b4",
        )

        # 趋势线
        z = np.polyfit(
            sentiment_lag.loc[valid.index].dropna(),
            valid.loc[sentiment_lag.loc[valid.index].dropna().index, "return"],
            1,
        )
        p = np.poly1d(z)
        x_line = np.linspace(
            sentiment_lag.loc[valid.index].min(),
            sentiment_lag.loc[valid.index].max(),
            100,
        )
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"趋势线 (slope={z[0]:.4f})")

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel(f"情感得分 (lag={lag})", fontsize=12)
        ax.set_ylabel("日收益率", fontsize=12)
        ax.set_title(f"滞后{lag}日情感 vs 当日收益", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save(fig, f"sentiment_vs_return_lag{lag}_{sentiment_col}")

    # ================================================================
    # VAR 脉冲响应图
    # ================================================================
    def plot_impulse_response(self, irf_data: dict, steps: int = 10):
        """绘制脉冲响应函数"""
        var_names = ["return", "sentiment", "volume"]
        n = len(var_names)

        fig, axes = plt.subplots(n, n, figsize=(15, 12))

        for i, cause in enumerate(var_names):
            for j, effect in enumerate(var_names):
                ax = axes[i][j]
                key = f"{cause}_→_{effect}"
                if key in irf_data:
                    values = irf_data[key][:steps]
                    ax.plot(range(steps), values, linewidth=2, color="#1f77b4")
                    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
                    ax.fill_between(range(steps), 0, values, alpha=0.1, color="#1f77b4")

                ax.set_title(f"{cause} → {effect}", fontsize=11)
                ax.set_xlabel("滞后天数", fontsize=9)
                if j == 0:
                    ax.set_ylabel("响应", fontsize=9)
                ax.grid(True, alpha=0.3)

        fig.suptitle("脉冲响应函数 (IRF)", fontsize=14, y=1.02)
        plt.tight_layout()

        return self._save(fig, "impulse_response")

    # ================================================================
    # 回归系数对比图
    # ================================================================
    def plot_regression_coefficients(
        self,
        regression_results: dict,
        title: str = "情感→收益回归系数",
    ):
        """绘制不同情感指标的回归系数对比"""
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = []
        coefs = []
        errors = []

        for name, result in regression_results.items():
            if "coefficients" in result and "results" in result:
                res = result["results"]
                for param_name in result["coefficients"].index:
                    if "sentiment" in param_name.lower() or "lag" in param_name.lower():
                        labels.append(f"{name}: {param_name}")
                        coefs.append(result["coefficients"][param_name])
                        # 95% 置信区间
                        conf = res.conf_int().loc[param_name]
                        errors.append([
                            result["coefficients"][param_name] - conf[0],
                            conf[1] - result["coefficients"][param_name],
                        ])

        if not labels:
            plt.close(fig)
            return None

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, coefs, xerr=np.array(errors).T, alpha=0.7, color="#1f77b4")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("系数值", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return self._save(fig, "regression_coefficients")

    # ================================================================
    # 情感分布图
    # ================================================================
    def plot_sentiment_distribution(self, df: pd.DataFrame, sentiment_col: str = "avg_sentiment"):
        """绘制情感得分分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        ax1.hist(df[sentiment_col].dropna(), bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax1.set_xlabel("情感得分", fontsize=12)
        ax1.set_ylabel("频数", fontsize=12)
        ax1.set_title(f"{sentiment_col} 分布", fontsize=13)

        # 箱线图（按月）
        df_copy = df.copy()
        df_copy["month"] = df_copy["trade_date"].dt.month
        monthly_data = [df_copy[df_copy["month"] == m][sentiment_col].dropna() for m in range(1, 13)]
        ax2.boxplot(monthly_data, labels=[f"{m}月" for m in range(1, 13)])
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("月份", fontsize=12)
        ax2.set_ylabel("情感得分", fontsize=12)
        ax2.set_title("月度情感分布", fontsize=13)

        plt.tight_layout()
        return self._save(fig, f"sentiment_distribution_{sentiment_col}")

    # ================================================================
    # 综合仪表板
    # ================================================================
    def plot_dashboard(self, df: pd.DataFrame, sentiment_col: str = "avg_sentiment"):
        """综合仪表板：行情 + 情感 + 交易量"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # 1. 收盘价
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df["trade_date"], df["close"], linewidth=1, color="#1f77b4")
        ax1.set_title("沪深300 收盘价", fontsize=13)
        ax1.set_ylabel("点位")
        ax1.grid(True, alpha=0.3)

        # 2. 情感指数
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df["trade_date"], df[sentiment_col], linewidth=0.8, color="#2ca02c")
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax2.set_title("日度情感指数", fontsize=12)
        ax2.set_ylabel("得分")
        ax2.grid(True, alpha=0.3)

        # 3. 新闻数量
        ax3 = fig.add_subplot(gs[1, 1])
        if "news_count" in df.columns:
            ax3.bar(df["trade_date"], df["news_count"], alpha=0.6, color="#ff7f0e", width=1)
        ax3.set_title("每日新闻数量", fontsize=12)
        ax3.set_ylabel("条数")
        ax3.grid(True, alpha=0.3)

        # 4. 情感 vs 收益散点
        ax4 = fig.add_subplot(gs[2, 0])
        valid = df.dropna(subset=[sentiment_col, "return"])
        ax4.scatter(valid[sentiment_col], valid["return"], alpha=0.2, s=10, color="#1f77b4")
        ax4.axhline(y=0, color="gray", alpha=0.3)
        ax4.axvline(x=0, color="gray", alpha=0.3)
        ax4.set_xlabel("情感得分")
        ax4.set_ylabel("收益率")
        ax4.set_title("情感 vs 收益", fontsize=12)
        ax4.grid(True, alpha=0.3)

        # 5. 交易量
        ax5 = fig.add_subplot(gs[2, 1])
        if "volume_detrend" in df.columns:
            ax5.plot(df["trade_date"], df["volume_detrend"], linewidth=0.8, color="#d62728")
        ax5.set_title("去趋势交易量", fontsize=12)
        ax5.set_ylabel("log(vol) - MA20")
        ax5.grid(True, alpha=0.3)

        # 格式化日期
        for ax in [ax1, ax2, ax3, ax5]:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

        return self._save(fig, "dashboard")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    viz = Visualizer()

    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n)
    df = pd.DataFrame({
        "trade_date": dates,
        "close": np.random.randn(n).cumsum() + 3500,
        "volume": np.random.randint(100000, 200000, n),
        "return": np.random.randn(n) * 0.01,
        "avg_sentiment": np.random.randn(n) * 0.3,
        "news_count": np.random.randint(0, 50, n),
        "volume_detrend": np.random.randn(n),
    })

    viz.plot_stock_price(df)
    viz.plot_sentiment_timeline(df)
    viz.plot_dashboard(df)
    print("可视化测试完成")
