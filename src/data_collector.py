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

from src.utils import get_tushare_token, load_config, PROJECT_ROOT

logger = logging.getLogger(__name__)


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
# AkShare 新闻采集（主力新闻源）
# ============================================================
class AkShareNewsCollector:
    """从 AkShare 获取财经新闻数据"""

    def __init__(self, config: dict):
        self.config = config["data"]
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]

    def fetch(self) -> pd.DataFrame:
        """
        获取财经新闻
        返回 DataFrame: date, title, content, source
        """
        start = datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.strptime(self.end_date, "%Y%m%d")

        all_news = []

        # 方法1: 东方财富快讯
        logger.info("尝试从东方财富获取财经新闻...")
        try:
            news_df = self._fetch_eastmoney_news(start, end)
            if news_df is not None and len(news_df) > 0:
                all_news.append(news_df)
                logger.info(f"东方财富: 获取 {len(news_df)} 条新闻")
        except Exception as e:
            logger.warning(f"东方财富新闻获取失败: {e}")

        # 方法2: 新浪财经新闻
        logger.info("尝试从新浪财经获取新闻...")
        try:
            news_df = self._fetch_sina_news(start, end)
            if news_df is not None and len(news_df) > 0:
                all_news.append(news_df)
                logger.info(f"新浪财经: 获取 {len(news_df)} 条新闻")
        except Exception as e:
            logger.warning(f"新浪财经新闻获取失败: {e}")

        if not all_news:
            logger.warning("AkShare 新闻源均未获取到数据，将尝试 GDELT")
            return pd.DataFrame()

        result = pd.concat(all_news, ignore_index=True)
        result = result.drop_duplicates(subset=["title"], keep="first")
        result = result.sort_values("date").reset_index(drop=True)

        logger.info(f"AkShare 共获取 {len(result)} 条去重新闻")
        return result

    def _fetch_eastmoney_news(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        """获取东方财富财经新闻"""
        try:
            # 使用 akshare 的财经新闻接口
            news_df = ak.stock_news_em(symbol="399300")
            if news_df is None or len(news_df) == 0:
                return None

            # 标准化列名
            col_map = {}
            for col in news_df.columns:
                col_lower = col.lower()
                if "时间" in col or "time" in col_lower or "date" in col_lower:
                    col_map[col] = "date"
                elif "标题" in col or "title" in col_lower:
                    col_map[col] = "title"
                elif "内容" in col or "content" in col_lower:
                    col_map[col] = "content"
                elif "来源" in col or "source" in col_lower:
                    col_map[col] = "source"

            news_df = news_df.rename(columns=col_map)

            if "date" in news_df.columns:
                news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
                news_df = news_df[
                    (news_df["date"] >= start) & (news_df["date"] <= end)
                ]

            if "source" not in news_df.columns:
                news_df["source"] = "东方财富"

            return news_df[["date", "title", "content", "source"]]
        except Exception as e:
            logger.debug(f"东方财富接口异常: {e}")
            return None

    def _fetch_sina_news(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        """获取新浪财经新闻"""
        try:
            # 使用 js_news 接口获取财经新闻
            news_df = ak.js_news(timestamp=start.strftime("%Y-%m-%d %H:%M:%S"))
            if news_df is None or len(news_df) == 0:
                return None

            col_map = {}
            for col in news_df.columns:
                col_lower = col.lower()
                if "时间" in col or "time" in col_lower or "date" in col_lower:
                    col_map[col] = "date"
                elif "标题" in col or "title" in col_lower or "新闻" in col:
                    col_map[col] = "title"
                elif "内容" in col or "content" in col_lower or "正文" in col:
                    col_map[col] = "content"

            news_df = news_df.rename(columns=col_map)

            if "date" in news_df.columns:
                news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
                news_df = news_df[
                    (news_df["date"] >= start) & (news_df["date"] <= end)
                ]

            if "source" not in news_df.columns:
                news_df["source"] = "新浪财经"
            if "content" not in news_df.columns:
                news_df["content"] = news_df.get("title", "")

            return news_df[["date", "title", "content", "source"]]
        except Exception as e:
            logger.debug(f"新浪财经接口异常: {e}")
            return None


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

    # 2. 获取新闻数据（优先 AkShare，备选 GDELT）
    news_df = pd.DataFrame()

    ak_collector = AkShareNewsCollector(config)
    news_df = ak_collector.fetch()

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
