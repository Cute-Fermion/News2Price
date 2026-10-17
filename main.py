"""
News2Price 主入口脚本
用法: python main.py [--skip-news] [--method dictionary|finbert|llm|both]
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config
from src.data_collector import collect_all_data
from src.news_processor import NewsProcessor
from src.sentiment_analyzer import run_sentiment_analysis
from src.feature_engineer import FeatureEngineer
from src.analysis import run_full_analysis
from src.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="News2Price: 基于新闻文本的A股股价预测")
    parser.add_argument(
        "--method",
        choices=["dictionary", "finbert", "llm", "both"],
        default="both",
        help="情感分析方法 (default: both)",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="跳过新闻采集（使用已有数据）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (default: config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("main")

    # 加载配置
    config = load_config(args.config)
    logger.info(f"目标: {config['data']['index_name']} ({config['data']['index_code']})")
    logger.info(f"日期: {config['data']['start_date']} ~ {config['data']['end_date']}")

    # Step 1: 数据采集
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: 数据采集")
    logger.info("=" * 60)
    data = collect_all_data(config)
    stock_df = data["stock"]
    news_df = data["news"]
    logger.info(f"行情: {len(stock_df)} 条, 新闻: {len(news_df)} 条")

    if news_df.empty:
        logger.warning("无新闻数据！请检查数据源配置。")
        return

    # Step 2: 文本处理
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 文本处理")
    logger.info("=" * 60)
    processor = NewsProcessor(config)
    news_df = processor.process_dataframe(news_df)

    # Step 3: 情感分析
    logger.info("\n" + "=" * 60)
    logger.info(f"Step 3: 情感分析 (method={args.method})")
    logger.info("=" * 60)
    news_df = run_sentiment_analysis(news_df, config, method=args.method)

    # Step 4: 特征工程
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: 特征工程")
    logger.info("=" * 60)
    fe = FeatureEngineer(config)
    merged_df = fe.build_features(stock_df, news_df, score_col="dict_score")

    # 保存处理后的数据
    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    merged_df.to_csv(
        os.path.join(processed_dir, "merged_dataset.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Step 5: 可视化
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: 可视化")
    logger.info("=" * 60)
    viz = Visualizer(config)
    viz.plot_stock_price(merged_df)
    viz.plot_sentiment_timeline(merged_df, "avg_sentiment")
    viz.plot_sentiment_vs_return(merged_df, "avg_sentiment", lag=1)
    viz.plot_sentiment_distribution(merged_df, "avg_sentiment")
    viz.plot_dashboard(merged_df, "avg_sentiment")

    # Step 6: 统计分析
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: 统计分析")
    logger.info("=" * 60)
    results = run_full_analysis(merged_df, config)

    # 脉冲响应图
    if "irf" in results:
        viz.plot_impulse_response(results["irf"]["orthogonal_irf"])

    logger.info("\n" + "=" * 60)
    logger.info("全部流程完成！")
    logger.info(f"结果保存在: {os.path.join(os.path.dirname(__file__), 'results')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
