"""
数据采集模块
- Tushare: 沪深300指数日频行情
- GDELT / AkShare: 财经新闻数据
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import tushare as ts
import akshare as ak

from src.utils import get_tushare_token, load_config, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)


# ============================================================
# Tushare 指数行情采集
# ============================================================
class TushareIndexCollector:
    """从 Tushare 获取指数日频行情数据"""

    def __init__(self, config: dict):
        self.config = config["data"]
        self.index_code = self.config["index_code"]
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.retry_times = self.config["tushare"]["retry_times"]
        self.retry_delay = self.config["tushare"]["retry_delay"]

        token = get_tushare_token()
        self.pro = ts.pro_api(token)

    def fetch(self) -> pd.DataFrame:
        """
        获取指数日线数据
        返回 DataFrame: trade_date, open, high, low, close, vol, amount, pct_chg
        """
        logger.info(
            f"从 Tushare 获取 {self.index_code} 行情 "
            f"({self.start_date} ~ {self.end_date})"
        )

        for attempt in range(1, self.retry_times + 1):
            try:
                df = self.pro.index_daily(
                    ts_code=self.index_code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )
                if df is not None and len(df) > 0:
                    break
            except Exception as e:
                logger.warning(f"第 {attempt} 次请求失败: {e}")
                if attempt < self.retry_times:
                    time.sleep(self.retry_delay)
        else:
            raise RuntimeError("Tushare 行情数据获取失败，请检查网络和 Token")

        # 标准化列名和类型
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = df.rename(columns={"vol": "volume"})

        logger.info(f"获取到 {len(df)} 条行情记录")
        return df[
            ["trade_date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
        ]


# ============================================================
# AkShare 新闻采集（央视财经新闻）
# ============================================================
class AkShareNewsCollector:
    """
    从 AkShare 的 news_cctv 接口获取央视财经新闻数据。
    按交易日逐日获取，每次调用返回当天约 10-15 条新闻。
    """

    def __init__(self, config: dict, trade_dates: list[str] | None = None):
        self.config = config["data"]
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.trade_dates = trade_dates  # 交易日列表 (YYYYMMDD 格式)
        self.delay = self.config.get("akshare", {}).get("delay", 0.3)

    def fetch(self) -> pd.DataFrame:
        """
        获取财经新闻
        返回 DataFrame: date, title, content, source
        """
        logger.info("通过 AkShare news_cctv 获取央视财经新闻...")

        # 确定要获取的日期列表：优先使用交易日，否则按周遍历
        if self.trade_dates:
            dates_to_fetch = self.trade_dates
        else:
            dates_to_fetch = self._generate_weekday_dates()

        all_news = []
        total = len(dates_to_fetch)
        failed = 0

        for i, date_str in enumerate(dates_to_fetch):
            try:
                df = ak.news_cctv(date=date_str)
                if df is not None and len(df) > 0:
                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
                    df["source"] = "央视新闻"
                    all_news.append(df[["date", "title", "content", "source"]])

                if (i + 1) % 50 == 0 or (i + 1) == total:
                    logger.info(f"  进度: {i+1}/{total}，已获取 {sum(len(x) for x in all_news)} 条新闻")
            except Exception as e:
                failed += 1
                if failed <= 3:
                    logger.debug(f"  {date_str} 获取失败: {e}")

            time.sleep(self.delay)

        if not all_news:
            logger.warning("AkShare 新闻源未获取到数据，将尝试 GDELT")
            return pd.DataFrame()

        result = pd.concat(all_news, ignore_index=True)
        result = result.dropna(subset=["title", "content"])
        result = result.drop_duplicates(subset=["title"], keep="first")
        result = result.sort_values("date").reset_index(drop=True)

        logger.info(f"AkShare 共获取 {len(result)} 条去重新闻（{failed} 个日期失败）")
        return result

    def _generate_weekday_dates(self) -> list[str]:
        """生成日期范围内所有周一到周五的日期（YYYYMMDD 格式）"""
        start = datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.strptime(self.end_date, "%Y%m%d")
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates


# ============================================================
# Tushare 本地新闻缓存加载器
# ============================================================
class TushareNewsLoader:
    """
    从本地缓存加载 Tushare 新闻数据（由 scripts/fetch_tushare_news.py 采集）
    合并 news（新浪财经快讯）、major_news（新闻通讯）、cctv_news（新闻联播）三个源。
    统一输出: date, title, content, source
    """

    def __init__(self, config: dict, cache_dir: str | Path | None = None):
        self.config = config["data"]
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "data" / "tushare_news"
        self.cache_dir = Path(cache_dir)
        # 过滤后数据目录（优先使用）
        self.filtered_dir = PROJECT_ROOT / "data" / "filtered_news"

    def is_available(self) -> bool:
        """检查本地缓存是否存在且有数据（优先检查过滤后数据）"""
        # 优先检查过滤后数据
        filtered_file = self.filtered_dir / "filtered_news.csv"
        if filtered_file.exists():
            return True
        # 回退到原始缓存
        if not self.cache_dir.exists():
            return False
        files = list(self.cache_dir.glob("*.csv"))
        return len(files) > 0

    def fetch(self) -> pd.DataFrame:
        """
        加载本地缓存并合并为统一格式
        优先使用过滤后数据 (data/filtered_news/filtered_news.csv)
        返回 DataFrame: date, title, content, source
        """
        # 优先检查过滤后数据
        filtered_file = self.filtered_dir / "filtered_news.csv"
        if filtered_file.exists():
            logger.info(f"加载过滤后新闻数据: {filtered_file}")
            result = pd.read_csv(filtered_file, parse_dates=["date"])
            # 过滤日期范围
            start = pd.to_datetime(self.start_date, format="%Y%m%d")
            end = pd.to_datetime(self.end_date, format="%Y%m%d")
            result = result[(result["date"] >= start) & (result["date"] <= end)]
            result = result.sort_values("date").reset_index(drop=True)
            logger.info(f"过滤后新闻数据: {len(result)} 条")
            return result

        logger.info(f"从本地缓存加载 Tushare 新闻: {self.cache_dir}")

        all_dfs = []

        # 1. 新浪财经快讯 (news_sina.csv)
        news_file = self.cache_dir / "news_sina.csv"
        if news_file.exists():
            df = pd.read_csv(news_file)
            if len(df) > 0:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df["date"] = df["datetime"].dt.normalize()
                df["source"] = "新浪财经"
                df = df.rename(columns={"title": "title", "content": "content"})
                all_dfs.append(df[["date", "title", "content", "source"]])
                logger.info(f"  新浪财经快讯: {len(df)} 条")

        # 2. 新闻通讯 (major_news.csv)
        major_file = self.cache_dir / "major_news.csv"
        if major_file.exists():
            df = pd.read_csv(major_file)
            if len(df) > 0:
                df["pub_time"] = pd.to_datetime(df["pub_time"], errors="coerce")
                df["date"] = df["pub_time"].dt.normalize()
                # source 字段已在数据中，统一列名
                if "src" in df.columns:
                    df["source"] = df["src"]
                else:
                    df["source"] = "新闻通讯"
                all_dfs.append(df[["date", "title", "content", "source"]])
                logger.info(f"  新闻通讯: {len(df)} 条")

        # 3. 新闻联播 (cctv_news.csv)
        cctv_file = self.cache_dir / "cctv_news.csv"
        if cctv_file.exists():
            df = pd.read_csv(cctv_file)
            if len(df) > 0:
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
                df["source"] = "央视新闻联播"
                all_dfs.append(df[["date", "title", "content", "source"]])
                logger.info(f"  新闻联播: {len(df)} 条")

        if not all_dfs:
            logger.warning("本地缓存中无有效数据")
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)

        # 过滤日期范围
        start = pd.to_datetime(self.start_date, format="%Y%m%d")
        end = pd.to_datetime(self.end_date, format="%Y%m%d")
        result = result[(result["date"] >= start) & (result["date"] <= end)]

        # 清洗
        result = result.dropna(subset=["title", "content"])
        result = result.drop_duplicates(subset=["date", "title"], keep="first")
        result = result.sort_values("date").reset_index(drop=True)

        logger.info(f"本地 Tushare 新闻合计: {len(result)} 条 (去重后)")
        return result


# ============================================================
# GDELT 新闻采集（备选/补充新闻源）
# ============================================================
class GDELTNewsCollector:
    """
    从 GDELT 获取新闻数据
    优先使用 BigQuery (pandas-gbq)，备选直接下载 GKG 文件
    """

    def __init__(self, config: dict):
        self.config = config["data"]["gdelt"]
        self.start_date = config["data"]["start_date"]
        self.end_date = config["data"]["end_date"]
        self.keywords = self.config.get("keywords", [])

    def fetch(self) -> pd.DataFrame:
        """获取 GDELT 新闻数据"""
        # 尝试 BigQuery
        try:
            return self._fetch_via_bigquery()
        except Exception as e:
            logger.info(f"BigQuery 不可用 ({e})，尝试直接下载 GKG 文件...")

        # 备选：直接下载
        return self._fetch_via_download()

    def _fetch_via_bigquery(self) -> pd.DataFrame:
        """通过 Google BigQuery 查询 GDELT GKG 数据"""
        from pandas_gbq import read_gbq

        start = self.start_date[:4] + "-" + self.start_date[4:6] + "-" + self.start_date[6:]
        end = self.end_date[:4] + "-" + self.end_date[4:6] + "-" + self.end_date[6:]

        # 构建关键词过滤条件
        theme_conditions = " OR ".join(
            [f"LOWER(themes) LIKE '%{kw.lower()}%'" for kw in self.keywords]
        )

        query = f"""
        SELECT
            DATE(date) as date,
            documentidentifier as url,
            themes,
            locations,
            persons,
            organizations,
            tone,
            SOURCEURL as source_url,
            SOURCECOMMONNAME as source_name
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE DATE(date) BETWEEN '{start}' AND '{end}'
          AND ({theme_conditions})
        ORDER BY date
        """

        logger.info("通过 BigQuery 查询 GDELT...")
        df = read_gbq(query, dialect="standard")
        df["date"] = pd.to_datetime(df["date"])

        logger.info(f"GDELT BigQuery 获取 {len(df)} 条新闻")
        return df

    def _fetch_via_download(self) -> pd.DataFrame:
        """
        直接下载 GDELT GKG 文件
        注意：此方法较慢，适合小日期范围
        """
        import requests
        import io
        import zipfile

        base_url = self.config["base_url"]
        start = datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.strptime(self.end_date, "%Y%m%d")

        all_records = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            # GKG 文件每天有两个（000000 和 010000 等）
            for suffix in ["000000", "010000"]:
                filename = f"{date_str}{suffix}.gkg.csv.zip"
                url = f"{base_url}/{filename}"

                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200:
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                            csv_name = zf.namelist()[0]
                            with zf.open(csv_name) as f:
                                # GKG 字段用 \t 分隔
                                records = pd.read_csv(
                                    f,
                                    sep="\t",
                                    header=None,
                                    on_bad_lines="skip",
                                    dtype=str,
                                )
                                # GKG 基本字段: GKGRECORDID, DATE, SourceCollectionIdentifier,
                                # SourceCommonName, DocumentIdentifier, Counts, Themes, Locations...
                                if len(records.columns) >= 13:
                                    records.columns = [
                                        "gkg_id", "date", "source_id",
                                        "source_name", "url", "counts",
                                        "themes", "locations", "persons",
                                        "organizations", "tone", "image_info",
                                        "social_image_shares",
                                    ] + [
                                        f"col_{i}"
                                        for i in range(13, len(records.columns))
                                    ]
                                    records["date_parsed"] = pd.to_datetime(
                                        records["date"], format="%Y%m%d%H%M%S", errors="coerce"
                                    )
                                    records["date_only"] = records["date_parsed"].dt.date
                                    all_records.append(records)
                except Exception as e:
                    logger.debug(f"下载 {filename} 失败: {e}")

            current += timedelta(days=1)
            # 避免请求过快
            time.sleep(0.5)

        if not all_records:
            logger.warning("GDELT 直接下载未获取到数据")
            return pd.DataFrame()

        df = pd.concat(all_records, ignore_index=True)

        # 关键词过滤
        if self.keywords:
            theme_pattern = "|".join(self.keywords)
            mask = df["themes"].fillna("").str.contains(
                theme_pattern, case=False, na=False
            )
            df = df[mask]

        df["date"] = pd.to_datetime(df["date_only"])
        df = df[["date", "url", "themes", "source_name", "tone"]].copy()
        df = df.rename(columns={"url": "title", "source_name": "source"})
        df["content"] = df["themes"]  # 用 themes 作为内容近似

        logger.info(f"GDELT 下载获取 {len(df)} 条新闻")
        return df


# ============================================================
# 统一数据加载入口
# ============================================================
def collect_all_data(config: dict | None = None) -> dict:
    """
    采集所有数据，返回字典:
    {
        "stock": pd.DataFrame,   # 行情数据
        "news": pd.DataFrame,    # 新闻数据
    }
    """
    if config is None:
        config = load_config()

    # 1. 获取行情数据
    stock_collector = TushareIndexCollector(config)
    stock_df = stock_collector.fetch()

    # 2. 获取新闻数据
    # 优先级: 过滤后新闻 > 本地 Tushare 缓存 > AkShare > GDELT
    trade_dates = stock_df["trade_date"].dt.strftime("%Y%m%d").tolist()

    news_df = pd.DataFrame()

    # 优先尝试本地 Tushare 缓存
    tushare_loader = TushareNewsLoader(config)
    if tushare_loader.is_available():
        logger.info("检测到本地 Tushare 新闻缓存，优先使用")
        news_df = tushare_loader.fetch()

    # 本地缓存不可用，回退到 AkShare
    if news_df.empty:
        logger.info("本地 Tushare 缓存不可用，尝试 AkShare...")
        ak_collector = AkShareNewsCollector(config, trade_dates=trade_dates)
        news_df = ak_collector.fetch()

    # AkShare 也没有，回退到 GDELT
    if news_df.empty:
        logger.info("AkShare 无数据，切换到 GDELT...")
        gdelt_collector = GDELTNewsCollector(config)
        news_df = gdelt_collector.fetch()

    if news_df.empty:
        logger.warning("未获取到任何新闻数据！后续情感分析将无法进行。")

    # 3. 保存到本地
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stock_df.to_csv(raw_dir / "stock_data.csv", index=False, encoding="utf-8-sig")
    if not news_df.empty:
        news_df.to_csv(raw_dir / "news_data.csv", index=False, encoding="utf-8-sig")

    return {"stock": stock_df, "news": news_df}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    data = collect_all_data()
    print(f"行情数据: {len(data['stock'])} 条")
    print(f"新闻数据: {len(data['news'])} 条")
