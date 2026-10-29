"""
新闻过滤模块
基于黑名单和白名单关键词过滤新闻数据
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils import PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)


class NewsFilter:
    """
    新闻过滤器：黑名单排除 + 白名单准入

    黑名单：命中任一关键词即丢弃（外国新闻、非财经新闻）
    白名单：至少命中 min_whitelist_hits 个关键词才保留（确保与A股相关）
    """

    def __init__(self, config: dict | None = None):
        """
        初始化过滤器

        参数:
            config: 配置字典，可包含 news_filter 节：
                - blacklist_path: 黑名单文件路径
                - whitelist_path: 白名单文件路径
                - min_whitelist_hits: 白名单最少命中数（默认2）
                - content_max_length: 内容匹配最大长度（默认200）
        """
        if config is None:
            from src.utils import load_config
            config = load_config()

        filter_config = config.get("news_filter", {})

        # 黑名单路径
        blacklist_path = filter_config.get("blacklist_path", "words/blacklist.txt")
        if not Path(blacklist_path).is_absolute():
            blacklist_path = PROJECT_ROOT / blacklist_path
        self.blacklist = self._load_words(blacklist_path)

        # 白名单路径
        whitelist_path = filter_config.get("whitelist_path", "words/whitelist.txt")
        if not Path(whitelist_path).is_absolute():
            whitelist_path = PROJECT_ROOT / whitelist_path
        self.whitelist = self._load_words(whitelist_path)

        # 过滤参数
        self.min_whitelist_hits = filter_config.get("min_whitelist_hits", 2)
        self.content_max_length = filter_config.get("content_max_length", 200)

        logger.info(f"新闻过滤器初始化完成: 黑名单 {len(self.blacklist)} 词, 白名单 {len(self.whitelist)} 词")
        logger.info(f"白名单最少命中数: {self.min_whitelist_hits}, 内容匹配长度: {self.content_max_length}")

    def _load_words(self, path: Path) -> list[str]:
        """加载词表文件（每行一个词，# 开头为注释）"""
        if not path.exists():
            logger.warning(f"词表文件不存在: {path}")
            return []

        words = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    words.append(line)
        return words

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤新闻 DataFrame

        输入输出列: date, title, content, source
        匹配范围: title 全文 + content 前 content_max_length 字符

        参数:
            df: 新闻 DataFrame

        返回:
            过滤后的 DataFrame
        """
        if df.empty:
            logger.warning("输入 DataFrame 为空，跳过过滤")
            return df

        original_count = len(df)
        logger.info(f"开始过滤: {original_count} 条新闻")

        # 构建匹配文本：title + content 前 N 字符
        title_text = df["title"].fillna("").astype(str)
        content_text = df["content"].fillna("").astype(str).str[:self.content_max_length]
        match_text = title_text + " " + content_text

        # 第一层：黑名单过滤（命中任一即丢弃）
        blacklist_mask = pd.Series([False] * len(df), index=df.index)
        for word in self.blacklist:
            blacklist_mask |= match_text.str.contains(word, regex=False, na=False)

        df_after_blacklist = df[~blacklist_mask].copy()
        blacklist_removed = original_count - len(df_after_blacklist)
        logger.info(f"黑名单过滤: 移除 {blacklist_removed} 条，剩余 {len(df_after_blacklist)} 条")

        if df_after_blacklist.empty:
            logger.warning("黑名单过滤后无剩余数据")
            return df_after_blacklist

        # 第二层：白名单过滤（至少命中 min_whitelist_hits 个）
        match_text_filtered = df_after_blacklist["title"].fillna("").astype(str) + " " + \
                              df_after_blacklist["content"].fillna("").astype(str).str[:self.content_max_length]

        whitelist_hits = pd.Series([0] * len(df_after_blacklist), index=df_after_blacklist.index)
        for word in self.whitelist:
            whitelist_hits += match_text_filtered.str.contains(word, regex=False, na=False).astype(int)

        df_final = df_after_blacklist[whitelist_hits >= self.min_whitelist_hits].copy()
        whitelist_removed = len(df_after_blacklist) - len(df_final)
        logger.info(f"白名单过滤: 移除 {whitelist_removed} 条，剩余 {len(df_final)} 条")
        logger.info(f"过滤完成: {original_count} -> {len(df_final)} 条 (保留率 {len(df_final)/original_count*100:.1f}%)")

        return df_final

    def get_filter_stats(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
        """
        获取过滤统计信息

        返回:
            {
                "original_count": int,
                "filtered_count": int,
                "retention_rate": float,
                "blacklist_removed": int,
                "whitelist_removed": int,
            }
        """
        return {
            "original_count": len(df_before),
            "filtered_count": len(df_after),
            "retention_rate": len(df_after) / len(df_before) if len(df_before) > 0 else 0.0,
        }
