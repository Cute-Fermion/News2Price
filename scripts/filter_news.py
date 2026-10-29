"""
新闻过滤脚本
读取 data/tushare_news/ 中的原始新闻数据，过滤后保存到 data/filtered_news/

用法:
    python scripts/filter_news.py [--config path] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

import pandas as pd

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, PROJECT_ROOT, setup_logger
from src.news_filter import NewsFilter

logger = setup_logger(__name__)


def load_raw_news(config: dict) -> pd.DataFrame:
    """
    从 data/tushare_news/ 加载原始新闻数据

    返回统一的 DataFrame: date, title, content, source
    """
    raw_dir = PROJECT_ROOT / "data" / "tushare_news"
    if not raw_dir.exists():
        logger.error(f"原始新闻目录不存在: {raw_dir}")
        return pd.DataFrame()

    all_dfs = []

    # 1. news_sina.csv (新浪财经快讯)
    news_sina = raw_dir / "news_sina.csv"
    if news_sina.exists():
        try:
            df = pd.read_csv(news_sina, low_memory=False)
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["source"] = "新浪财经"
            df = df[["date", "title", "content", "source"]]
            all_dfs.append(df)
            logger.info(f"加载 news_sina.csv: {len(df)} 条")
        except Exception as e:
            logger.warning(f"加载 news_sina.csv 失败: {e}")

    # 2. major_news.csv (新闻通讯)
    major_news = raw_dir / "major_news.csv"
    if major_news.exists():
        try:
            df = pd.read_csv(major_news)
            df["date"] = pd.to_datetime(df["pub_time"], errors="coerce")
            df["source"] = df.get("src", "新闻通讯")
            df = df[["date", "title", "content", "source"]]
            all_dfs.append(df)
            logger.info(f"加载 major_news.csv: {len(df)} 条")
        except Exception as e:
            logger.warning(f"加载 major_news.csv 失败: {e}")

    # 3. cctv_news.csv (新闻联播文字稿)
    cctv_news = raw_dir / "cctv_news.csv"
    if cctv_news.exists():
        try:
            df = pd.read_csv(cctv_news)
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df["source"] = "央视新闻联播"
            df = df[["date", "title", "content", "source"]]
            all_dfs.append(df)
            logger.info(f"加载 cctv_news.csv: {len(df)} 条")
        except Exception as e:
            logger.warning(f"加载 cctv_news.csv 失败: {e}")

    if not all_dfs:
        logger.error("未加载到任何新闻数据")
        return pd.DataFrame()

    # 合并所有来源
    result = pd.concat(all_dfs, ignore_index=True)

    # 按日期范围过滤
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"])
    result = result[(result["date"] >= start_date) & (result["date"] <= end_date)]

    # 去重
    result = result.drop_duplicates(subset=["date", "title"], keep="first")
    result = result.sort_values("date").reset_index(drop=True)

    logger.info(f"合并后共 {len(result)} 条新闻 (日期范围: {start_date} ~ {end_date})")
    return result


def main():
    parser = argparse.ArgumentParser(description="新闻过滤脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--force", action="store_true", help="强制重新过滤（覆盖已有结果）")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 输出目录
    filtered_dir = PROJECT_ROOT / "data" / "filtered_news"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    output_file = filtered_dir / "filtered_news.csv"

    # 检查是否已有过滤结果
    if output_file.exists() and not args.force:
        logger.info(f"已存在过滤结果: {output_file}")
        logger.info("使用 --force 参数强制重新过滤")
        return

    # 加载原始数据
    logger.info("=" * 60)
    logger.info("Step 1: 加载原始新闻数据")
    logger.info("=" * 60)
    raw_df = load_raw_news(config)
    if raw_df.empty:
        logger.error("无原始数据，退出")
        return

    # 过滤
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 应用黑名单 + 白名单过滤")
    logger.info("=" * 60)
    news_filter = NewsFilter(config)
    filtered_df = news_filter.filter_dataframe(raw_df)

    if filtered_df.empty:
        logger.error("过滤后无数据，请检查词表配置")
        return

    # 保存结果
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: 保存过滤结果")
    logger.info("=" * 60)
    filtered_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"过滤结果已保存: {output_file}")
    logger.info(f"  原始: {len(raw_df)} 条")
    logger.info(f"  过滤后: {len(filtered_df)} 条")
    logger.info(f"  保留率: {len(filtered_df)/len(raw_df)*100:.1f}%")

    # 按来源统计
    source_counts = filtered_df["source"].value_counts()
    logger.info("\n按来源统计:")
    for source, count in source_counts.items():
        logger.info(f"  {source}: {count} 条")

    # 按月份统计（每月）
    filtered_df["month"] = filtered_df["date"].dt.to_period("M")
    monthly_counts = filtered_df["month"].value_counts().sort_index()
    logger.info("\n按月份统计 (前10个月):")
    for month, count in list(monthly_counts.items())[:10]:
        logger.info(f"  {month}: {count} 条")

    # 按日统计
    logger.info("\n" + "=" * 60)
    logger.info("按日统计")
    logger.info("=" * 60)
    filtered_df["date_only"] = filtered_df["date"].dt.date
    daily_counts = filtered_df.groupby("date_only").size()

    # 生成完整日期范围
    start = pd.to_datetime(config["data"]["start_date"])
    end = pd.to_datetime(config["data"]["end_date"])
    all_dates = pd.date_range(start, end, freq="D")
    all_date_set = set(d.date() for d in all_dates)

    # 找出有新闻的日期
    dates_with_news = set(daily_counts.index)
    dates_without_news = sorted(all_date_set - dates_with_news)

    logger.info(f"日期范围: {config['data']['start_date']} ~ {config['data']['end_date']}")
    logger.info(f"总天数: {len(all_dates)}")
    logger.info(f"有新闻的天数: {len(dates_with_news)}")
    logger.info(f"无新闻的天数: {len(dates_without_news)}")

    if dates_without_news:
        logger.info(f"\n无新闻的日期 (共 {len(dates_without_news)} 天):")
        for d in dates_without_news[:50]:  # 只显示前50个
            logger.info(f"  {d}")
        if len(dates_without_news) > 50:
            logger.info(f"  ... 还有 {len(dates_without_news) - 50} 天")

    # 每日新闻数分布
    logger.info(f"\n每日新闻数分布:")
    logger.info(f"  最小: {daily_counts.min() if len(daily_counts) > 0 else 0}")
    logger.info(f"  最大: {daily_counts.max() if len(daily_counts) > 0 else 0}")
    logger.info(f"  中位数: {daily_counts.median():.0f}" if len(daily_counts) > 0 else "  中位数: 0")
    logger.info(f"  平均: {daily_counts.mean():.1f}" if len(daily_counts) > 0 else "  平均: 0")

    logger.info("\n" + "=" * 60)
    logger.info("过滤完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
