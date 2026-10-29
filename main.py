"""
News2Price 主入口脚本
用法: python main.py [--skip-news] [--method dictionary|finbert|llm|both] [--config path]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import os

import pandas as pd

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_config, setup_logger, init_run, get_run_dir,
    get_output_path, get_data_path, ensure_dir,
)
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
        default=None,
        help="情感分析方法 (default: None)",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="跳过新闻采集（使用已有数据）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径 (default: config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新分析，忽略已有的 LLM 断点文件 (data/llm_temp/llm_checkpoint.csv)",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 初始化运行目录（自动递增序号 + 时间戳）
    run_id = init_run(config)

    # 为所有子模块补充文件 handler（模块在 import 时已创建 logger，但那时 run 目录尚未初始化）
    for mod_name in ["main", "src.data_collector", "src.news_processor",
                      "src.news_filter", "src.sentiment_analyzer", "src.feature_engineer",
                      "src.analysis", "src.visualization"]:
        setup_logger(mod_name)

    # 配置日志（同时输出到控制台和 run.log）
    logger = setup_logger("main")
    logger.info(f"运行ID: {run_id}")
    logger.info(f"目标: {config['data']['index_name']} ({config['data']['index_code']})")
    logger.info(f"日期: {config['data']['start_date']} ~ {config['data']['end_date']}")

    # 确定输出路径
    figures_dir = get_output_path("figures")
    ensure_dir(figures_dir)

    # Step 1: 数据采集
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: 数据采集")
    logger.info("=" * 60)

    if args.skip_news:
        raw_dir = get_data_path("raw")
        filtered_dir = get_data_path("filtered_news")
        shutil.copy2(filtered_dir / "filtered_news.csv", raw_dir / "news_data.csv")
        stock_path = raw_dir / "stock_data.csv"
        news_path = raw_dir / "news_data.csv"

        stock_df = pd.read_csv(stock_path, parse_dates=["trade_date"])
        if news_path.exists():
            news_df = pd.read_csv(news_path, parse_dates=["date"])
            logger.info(f"从本地加载: 行情 {len(stock_df)} 条, 新闻 {len(news_df)} 条")
        else:
            news_df = pd.DataFrame()
            logger.warning("未找到本地新闻数据文件")
    else:
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
    if args.method is None:
        method = config.get("sentiment", {}).get("method", "dictionary")
    else:
        method = args.method
    logger.info(f"Step 3: 情感分析 (method={method})")
    logger.info("=" * 60)
    news_df = run_sentiment_analysis(news_df, config, method=method, force=args.force)
    news_df.to_csv(get_data_path("processed") / f"news_sentiment_{method}.csv", index=False, encoding="utf-8-sig")
    logger.info(f"情感分析结果已保存: {get_data_path('processed') / f'news_sentiment_{method}.csv'}")

    # Step 4: 特征工程
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: 特征工程")
    logger.info("=" * 60)
    # 根据方法选择情感得分列
    score_col_map = {
        "dictionary": "dict_score",
        "finbert": "finbert_score",
        "llm": "llm_score",
        "both": "dict_score",  # both 方法时默认用词典分数
    }
    score_col = score_col_map.get(method, "dict_score")
    logger.info(f"使用情感得分列: {score_col}")
    fe = FeatureEngineer(config)
    merged_df = fe.build_features(stock_df, news_df, score_col=score_col)

    # 保存处理后的数据（同时保存到 run 目录和原始目录）
    processed_dir = get_data_path("processed")
    ensure_dir(processed_dir)
    merged_df.to_csv(
        processed_dir / "merged_dataset.csv",
        index=False, encoding="utf-8-sig",
    )
    # 也复制一份到 run 目录
    run_dir = get_run_dir()
    if run_dir is not None:
        run_processed = run_dir / "processed"
        ensure_dir(run_processed)
        merged_df.to_csv(
            run_processed / "merged_dataset.csv",
            index=False, encoding="utf-8-sig",
        )

    # Step 5: 可视化
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: 可视化")
    logger.info("=" * 60)
    viz = Visualizer(config, output_dir=figures_dir)
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
    logger.info(f"运行目录: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
