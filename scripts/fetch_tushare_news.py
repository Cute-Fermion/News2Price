"""
Tushare 新闻数据批量采集脚本
用途：在限时高积分 token 有效期内，一次性拉取所有历史新闻数据存本地

采集内容：
1. news (新浪财经快讯) - 短讯，单次最多 1500 条
2. major_news (新闻通讯) - 长篇通讯，单次最多 800 条
3. cctv_news (新闻联播文字稿) - 每天约 12 条

数据范围：从 config.yaml 读取（当前: 20220101 ~ 20260831）
存储路径：data/tushare_news/

使用方法：
    # 1. 确保 .env 中的 TUSHARE_TOKEN 对应高积分账号（需要 5000+ 积分）
    # 2. 运行采集脚本（默认从 config.yaml 读取日期范围）
    python scripts/fetch_tushare_news.py
    
    # 或者指定日期范围
    python scripts/fetch_tushare_news.py --start 20230101 --end 20231231
    
    # 只采集某个接口
    python scripts/fetch_tushare_news.py --only news
    python scripts/fetch_tushare_news.py --only major_news
    python scripts/fetch_tushare_news.py --only cctv_news
    
    # 3. 采集完成后，主流程会自动从本地缓存加载（无需额外配置）
    python main.py --skip-news
"""
from __future__ import annotations

import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts
from Demos.BackupRead_BackupWrite import tempfile

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import get_tushare_token, load_config, setup_logger

logger = setup_logger(__name__)

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = PROJECT_ROOT / "data" / "tushare_news"

# 每次 API 调用之间的间隔（秒），避免触发限频
API_DELAY = 0.15  # ~400 calls/min，留点余量

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒


def get_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    """获取日期范围内的所有交易日"""
    logger.info(f"获取交易日列表: {start_date} ~ {end_date}")
    df = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    dates = df["cal_date"].tolist()
    logger.info(f"共 {len(dates)} 个交易日")
    return dates


# ============================================================
# 1. 新浪财经快讯 (news)
# ============================================================
def fetch_news(pro, start_date: str, end_date: str, output_dir: Path) -> Path | None:
    """
    采集新浪财经快讯
    接口：news(src='sina', start_date=..., end_date=...)
    返回字段：datetime, content, title
    可选字段：score, channels
    
    每个 batch 写入一次文件，防止中断导致数据丢失
    """
    logger.info("=" * 60)
    logger.info("采集新浪财经快讯 (news)")
    logger.info("=" * 60)

    output_file = output_dir / "news_sina.csv"
    temp_file = output_dir / "news_sina_temp.csv"

    # 断点续采
    # 先读正式文件
    existing_df = pd.DataFrame()
    if output_file.exists() :
        existing_df = pd.read_csv(output_file, parse_dates=["datetime"])

    # 再读临时文件，合并
    if temp_file.exists() :
        temp_df = pd.read_csv(temp_file)
        if "datetime" in temp_df.columns :
            temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], errors="coerce")
        # 把解析失败（NaT）的行挑出来看看，通常就是错位行
        bad = temp_df[temp_df["datetime"].isna()]
        parsed = pd.to_datetime(temp_df["datetime"], errors="coerce")
        bad_mask = parsed.isna()
        bad_idx = temp_df.index[bad_mask].tolist()
        clean_df = temp_df.loc[~bad_mask].copy()
        print(f"删除 {len(temp_df) - len(clean_df)} 行，剩余 {len(clean_df)} 行")
        temp_df = clean_df
        if len(bad) :
            logger.warning(f"临时文件中有 {len(bad)} 行 datetime 解析失败，示例：\n{bad.head(3)}")
        if len(temp_df) > 0 :
            existing_df = pd.concat([existing_df, temp_df], ignore_index=True)


    # 合并完之后，统一根据 existing_df 算续采起点
    if len(existing_df) > 0 :
        last_day = pd.to_datetime(existing_df["datetime"]).dt.date.max()
        start_date = (last_day + timedelta(days=1)).strftime("%Y%m%d")
        logger.info(f"已有 {len(existing_df)} 条，从 {start_date} 继续采集")

    current_start = start_date
    batch_size = 7  # 每次查 7 天
    total_collected = len(existing_df)
    batch_count = 0

    while current_start <= end_date:
        current_end = min(
            (datetime.strptime(current_start, "%Y%m%d") + timedelta(days=batch_size - 1)).strftime("%Y%m%d"),
            end_date,
        )

        batch_df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = pro.news(
                    src="sina",
                    start_date=f"{current_start[:4]}-{current_start[4 :6]}-{current_start[6 :]} 00:00:00",
                    end_date=f"{current_end[:4]}-{current_end[4 :6]}-{current_end[6 :]} 23:59:59",
                    fields="datetime,content,title",
                )
                if df is not None and len(df) > 0:
                    batch_df = df
                    logger.info(f"  {current_start}~{current_end}: {len(df)} 条")
                else:
                    logger.debug(f"  {current_start}~{current_end}: 无数据")
                break
            except Exception as e:
                logger.warning(f"  {current_start}~{current_end} 第{attempt}次失败: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"  {current_start}~{current_end} 采集失败，跳过")

        # 每个 batch 写入一次（追加到临时文件）
        if batch_df is not None and len(batch_df) > 0:
            batch_count += 1
            total_collected += len(batch_df)
            
            # 写入临时文件（追加模式）
            if batch_count == 1:
                # 第一个 batch，创建新文件
                batch_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
            else:
                # 后续 batch，追加
                batch_df.to_csv(temp_file, mode="a", header=False, index=False, encoding="utf-8-sig")
            
            logger.info(f"    [已写入临时文件，累计 {total_collected} 条]")

        current_start = (datetime.strptime(current_end, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        time.sleep(API_DELAY)

    # 最终合并：已有数据 + 临时文件数据
    if temp_file.exists():
        try:
            temp_df = pd.read_csv(temp_file, parse_dates=["datetime"])
            if len(temp_df) > 0:
                existing_df = pd.concat([existing_df, temp_df], ignore_index=True)

        except Exception as e:
            logger.error(f"读取临时文件失败: {e}")
            return None

    if len(existing_df) == 0:
        logger.warning("news 接口未获取到数据")
        return None

    result = existing_df

    # 去重和排序
    result = result.drop_duplicates(subset=["datetime", "title"], keep="first")
    result = result.sort_values("datetime").reset_index(drop=True)

    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"新浪财经快讯保存至: {output_file} ({len(result)} 条)")
    return output_file


# ============================================================
# 2. 新闻通讯 (major_news)
# ============================================================
def fetch_major_news(pro, start_date: str, end_date: str, output_dir: Path) -> Path | None:
    """
    采集新闻通讯（长篇）
    接口：major_news(start_date=..., end_date=...)
    返回字段：title, pub_time, src, url
    可选字段：content
    
    每个 batch 写入一次文件，防止中断导致数据丢失
    """
    logger.info("=" * 60)
    logger.info("采集新闻通讯 (major_news)")
    logger.info("=" * 60)

    output_file = output_dir / "major_news.csv"
    temp_file = output_dir / "major_news_temp.csv"

    # 断点续采
    # 先读正式文件
    existing_df = pd.DataFrame()
    if output_file.exists() :
        existing_df = pd.read_csv(output_file, parse_dates=["pub_time"])

    # 再读临时文件，合并
    if temp_file.exists() :
        temp_df = pd.read_csv(temp_file, parse_dates=["pub_time"])
        if len(temp_df) > 0 :
            existing_df = pd.concat([existing_df, temp_df], ignore_index=True)
        temp_file.unlink()

    # 合并完之后，统一根据 existing_df 算续采起点
    if len(existing_df) > 0 :
        last_day = pd.to_datetime(existing_df["pub_time"]).dt.date.max()
        start_date = (last_day + timedelta(days=1)).strftime("%Y%m%d")
        logger.info(f"已有 {len(existing_df)} 条，从 {start_date} 继续采集")

    current_start = start_date
    batch_size = 10
    total_collected = len(existing_df)
    batch_count = 0

    while current_start <= end_date:
        current_end = min(
            (datetime.strptime(current_start, "%Y%m%d") + timedelta(days=batch_size - 1)).strftime("%Y%m%d"),
            end_date,
        )

        batch_df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = pro.major_news(
                    start_date=f"{current_start[:4]}-{current_start[4 :6]}-{current_start[6 :]} 00:00:00",
                    end_date=f"{current_end[:4]}-{current_end[4 :6]}-{current_end[6 :]} 23:59:59",
                    fields="title,pub_time,src,content",
                )
                if df is not None and len(df) > 0:
                    batch_df = df
                    logger.info(f"  {current_start}~{current_end}: {len(df)} 条")
                else:
                    logger.debug(f"  {current_start}~{current_end}: 无数据")
                break
            except Exception as e:
                logger.warning(f"  {current_start}~{current_end} 第{attempt}次失败: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"  {current_start}~{current_end} 采集失败，跳过")

        # 每个 batch 写入一次（追加到临时文件）
        if batch_df is not None and len(batch_df) > 0:
            batch_count += 1
            total_collected += len(batch_df)
            
            if batch_count == 1:
                batch_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
            else:
                batch_df.to_csv(temp_file, mode="a", header=False, index=False, encoding="utf-8-sig")
            
            logger.info(f"    [已写入临时文件，累计 {total_collected} 条]")

        current_start = (datetime.strptime(current_end, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        time.sleep(API_DELAY)

    # 最终合并：已有数据 + 临时文件数据
    if temp_file.exists():
        try:
            temp_df = pd.read_csv(temp_file, parse_dates=["pub_time"])
            if len(temp_df) > 0:
                existing_df = pd.concat([existing_df, temp_df], ignore_index=True)
            temp_file.unlink()
        except Exception as e:
            logger.error(f"读取临时文件失败: {e}")
            return None

    if len(existing_df) == 0:
        logger.warning("major_news 接口未获取到数据")
        return None

    result = existing_df

    result = result.drop_duplicates(subset=["pub_time", "title"], keep="first")
    result = result.sort_values("pub_time").reset_index(drop=True)

    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"新闻通讯保存至: {output_file} ({len(result)} 条)")
    return output_file


# ============================================================
# 3. 新闻联播文字稿 (cctv_news)
# ============================================================
def fetch_cctv_news(pro, start_date: str, end_date: str, output_dir: Path) -> Path | None:
    """
    采集新闻联播文字稿
    接口：cctv_news(date=...)
    返回字段：date, title, content
    注意：此接口按单天查询，需要逐天调用
    
    每 100 天写入一次文件，防止中断导致数据丢失
    """
    logger.info("=" * 60)
    logger.info("采集新闻联播文字稿 (cctv_news)")
    logger.info("=" * 60)

    output_file = output_dir / "cctv_news.csv"
    temp_file = output_dir / "cctv_news_temp.csv"

    # 断点续采
    existing_df = pd.DataFrame()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
        except Exception as e:
            logger.warning(f"读取已有数据失败: {e}")

    # 如果有临时文件，加载它
    if temp_file.exists():
        try:
            temp_df = pd.read_csv(temp_file)
            if len(temp_df) > 0:
                existing_df = pd.concat([existing_df, temp_df], ignore_index=True)
                logger.info(f"从临时文件恢复 {len(temp_df)} 条数据")
            temp_file.unlink()
        except Exception as e:
            logger.warning(f"读取临时文件失败: {e}")

    if len(existing_df) > 0 :
        last_day = pd.to_datetime(existing_df["date"]).dt.date.max()
        start_date = (datetime.strptime(str(last_day), "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        logger.info(f"已有 {len(existing_df)} 条，从 {start_date} 继续采集")

    # 生成需要采集的日期列表
    dates = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    batch_buffer = []
    failed_dates = []
    total_collected = len(existing_df)
    batch_count = 0

    for i, date in enumerate(dates):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = pro.cctv_news(date=date)
                if df is not None and len(df) > 0:
                    batch_buffer.append(df)
                    total_collected += len(df)
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    failed_dates.append(date)
                    logger.debug(f"  {date} 采集失败: {e}")

        # 每 100 天或缓冲区满 500 条时写入一次
        if (i + 1) % 100 == 0 or total_collected - batch_count * 500 >= 500:
            if batch_buffer:
                batch_count += 1
                batch_df = pd.concat(batch_buffer, ignore_index=True)
                
                if batch_count == 1:
                    batch_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
                else:
                    batch_df.to_csv(temp_file, mode="a", header=False, index=False, encoding="utf-8-sig")
                
                logger.info(f"  [已写入临时文件，累计 {total_collected} 条]")
                batch_buffer = []

        # 进度输出
        if (i + 1) % 100 == 0 or (i + 1) == len(dates):
            logger.info(f"  进度: {i+1}/{len(dates)}，已获取 {total_collected} 条")

        time.sleep(API_DELAY)

    # 写入剩余的缓冲区数据
    if batch_buffer:
        batch_count += 1
        batch_df = pd.concat(batch_buffer, ignore_index=True)
        
        if batch_count == 1:
            batch_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
        else:
            batch_df.to_csv(temp_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    # 最终合并：已有数据 + 临时文件数据
    if temp_file.exists():
        try:
            temp_df = pd.read_csv(temp_file)
            if len(temp_df) > 0:
                existing_df = pd.concat([existing_df, temp_df], ignore_index=True)
            temp_file.unlink()
        except Exception as e:
            logger.error(f"读取临时文件失败: {e}")
            return None

    if len(existing_df) == 0:
        logger.warning("cctv_news 接口未获取到数据")
        return None

    result = existing_df

    result = result.drop_duplicates(subset=["date", "title"], keep="first")
    result = result.sort_values("date").reset_index(drop=True)

    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"新闻联播保存至: {output_file} ({len(result)} 条)")
    if failed_dates:
        logger.warning(f"  失败日期: {len(failed_dates)} 天")
    return output_file


# ============================================================
# 主流程
# ============================================================
def main():
    # 从 config.yaml 读取日期范围
    config = load_config()
    default_start = config["data"]["start_date"]
    default_end = config["data"]["end_date"]

    parser = argparse.ArgumentParser(description="Tushare 新闻数据批量采集")
    parser.add_argument("--start", default=default_start, help=f"开始日期 (YYYYMMDD)，默认从 config.yaml 读取: {default_start}")
    parser.add_argument("--end", default=default_end, help=f"结束日期 (YYYYMMDD)，默认从 config.yaml 读取: {default_end}")
    parser.add_argument("--only", choices=["news", "major_news", "cctv_news"], help="只采集指定接口")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 初始化
    token = get_tushare_token()
    if not token:
        logger.error("未找到 TUSHARE_TOKEN，请检查 .env 文件")
        sys.exit(1)

    pro = ts.pro_api(token)
    pro._DataApi__http_url = "https://tushare.citydata.club"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tushare 新闻数据批量采集")
    logger.info(f"日期范围: {args.start} ~ {args.end}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"API 延迟设置: {API_DELAY}s (约 {int(60/API_DELAY)} calls/min)")

    start_time = time.time()
    results = {}

    # 采集
    if args.only is None or args.only == "news":
        try:
            results["news"] = fetch_news(pro, args.start, args.end, output_dir)
        except Exception as e:
            logger.error(f"news 采集异常: {e}")
            results["news"] = None

    if args.only is None or args.only == "major_news":
        try:
            results["major_news"] = fetch_major_news(pro, args.start, args.end, output_dir)
        except Exception as e:
            logger.error(f"major_news 采集异常: {e}")
            results["major_news"] = None

    if args.only is None or args.only == "cctv_news":
        try:
            results["cctv_news"] = fetch_cctv_news(pro, args.start, args.end, output_dir)
        except Exception as e:
            logger.error(f"cctv_news 采集异常: {e}")
            results["cctv_news"] = None

    elapsed = time.time() - start_time

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("采集完成汇总")
    logger.info("=" * 60)
    logger.info(f"耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

    for name, path in results.items():
        if path and Path(path).exists():
            df = pd.read_csv(path)
            logger.info(f"  {name}: {len(df)} 条 -> {path}")
        else:
            logger.info(f"  {name}: 无数据或采集失败")

    logger.info("")
    logger.info("后续步骤:")
    logger.info("  1. 检查 data/tushare_news/ 目录下的 CSV 文件")
    logger.info("  2. 修改 config.yaml 中的新闻源配置指向本地文件")
    logger.info("  3. 运行主流程测试: python main.py --skip-news")


if __name__ == "__main__":
    main()
