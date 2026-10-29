"""
新闻文本处理模块
- 文本清洗（HTML标签、特殊字符、空白符）
- 中文分词（jieba）
- 停用词过滤
- GDELT 字段解析
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

import jieba
import pandas as pd

from src.utils import load_config, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)

# 默认中文停用词（常用）
_DEFAULT_STOPWORDS = set(
    """
    的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着
    没有 看 好 自己 这 他 她 它 们 那 些 什么 因为 所以 如果 虽然 但是
    而 且 或 与 及 等 之 乎 者 也 矣 焉 哉 兮 啊 呀 呢 吧 吗 哦 嗯
    这个 那个 这些 那些 这样 那样 这里 那里 怎么 怎样 如何 为什么
    已经 还是 或者 以及 并且 而且 不过 只是 但是 然而 因此 所以
    对于 关于 根据 按照 由于 通过 经过 为了 对于 关于
    万 亿 元 美元 人民币 港元 日元 英镑 欧元
    据 记者 报道 讯 电 新华社 人民日报 央视 本报记者
    """.split()
)


class NewsProcessor:
    """新闻文本处理器"""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        tp_config = config.get("text_processing", {})
        self.remove_stopwords = tp_config.get("remove_stopwords", True)
        self.stopwords_path = tp_config.get("stopwords_path", "")

        # 加载停用词
        self.stopwords = set(_DEFAULT_STOPWORDS)
        if self.stopwords_path:
            # 支持相对路径（相对于项目根目录）
            sw_path = Path(self.stopwords_path)
            if not sw_path.is_absolute():
                sw_path = PROJECT_ROOT / sw_path
            if sw_path.exists():
                with open(sw_path, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip()
                        if word and not word.startswith("#"):
                            self.stopwords.add(word)
                logger.info(f"加载自定义停用词表: {sw_path}，共 {len(self.stopwords)} 个")
            else:
                logger.warning(f"停用词表文件不存在: {sw_path}")

        # 初始化 jieba
        jieba.setLogLevel(logging.WARNING)

    def clean_text(self, text: str) -> str:
        """
        清洗文本：
        - 去除 HTML 标签
        - 去除 URL
        - 去除特殊字符（保留中文、英文、数字、基本标点）
        - 规范化空白符
        """
        if not isinstance(text, str):
            return ""

        # 去除 HTML 标签
        text = re.sub(r"<[^>]+>", " ", text)

        # 去除 URL
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)

        # 去除 @提及 和 #话题#
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#.+?#", " ", text)

        # 去除多余特殊字符（保留中文、英文字母、数字、基本标点）
        text = re.sub(r"[^\w\s\u4e00-\u9fff，。！？、；：""''（）【】《》\-\.\,\!\?\;\:\(\)\[\]]", " ", text)

        # 规范化空白符
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """中文分词"""
        return list(jieba.cut(text, cut_all=False))

    def remove_stopwords_from_tokens(self, tokens: list[str]) -> list[str]:
        """去除停用词"""
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords and len(t.strip()) > 0]

    def process_text(self, text: str) -> dict:
        """
        完整文本处理流程
        返回: {
            "cleaned": 清洗后文本,
            "tokens": 分词结果,
            "tokens_filtered": 去停用词后的分词,
        }
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens_filtered = self.remove_stopwords_from_tokens(tokens)

        return {
            "cleaned": cleaned,
            "tokens": tokens,
            "tokens_filtered": tokens_filtered,
        }

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
    ) -> pd.DataFrame:
        """
        批量处理 DataFrame 中的新闻文本
        新增列: cleaned_text, tokens, tokens_filtered
        当 text_col 为空时，自动回退到 title 列
        """
        df = df.copy()

        # 确保文本列存在
        if text_col not in df.columns:
            # 尝试用 title 作为替代
            if "title" in df.columns:
                text_col = "title"
            else:
                logger.error(f"找不到文本列 '{text_col}'，可用列: {list(df.columns)}")
                return df

        # 填充空值
        df[text_col] = df[text_col].fillna("").astype(str)

        # 如果 content 列存在但大量为空，回退到 title
        if text_col == "content" and "title" in df.columns:
            empty_mask = df[text_col].str.strip() == ""
            if empty_mask.sum() > 0:
                logger.info(f"content 列有 {empty_mask.sum()} 条为空，使用 title 列补充")
                df.loc[empty_mask, text_col] = df.loc[empty_mask, "title"].fillna("").astype(str)

        # 逐行处理
        results = df[text_col].apply(self.process_text)
        df["cleaned_text"] = results.apply(lambda x: x["cleaned"])
        df["tokens"] = results.apply(lambda x: x["tokens"])
        df["tokens_filtered"] = results.apply(lambda x: x["tokens_filtered"])

        logger.info(f"处理完成: {len(df)} 条新闻")
        return df

    def parse_gdelt_themes(self, themes_str: str) -> list[str]:
        """
        解析 GDELT GKG 的 Themes 字段
        Themes 格式: "主题1;主题2;主题3" 或 "主题1,主题2"
        """
        if not isinstance(themes_str, str):
            return []
        # GDELT themes 用分号分隔
        themes = [t.strip() for t in themes_str.split(";") if t.strip()]
        return themes

    def parse_gdelt_tone(self, tone_str: str) -> dict:
        """
        解析 GDELT GKG 的 Tone 字段
        Tone 格式: "avgTone,avgTone*10,avgTone*100"
        返回: {"avg_tone": float, "positive": float, "negative": float}
        """
        if not isinstance(tone_str, str):
            return {"avg_tone": 0.0, "positive": 0.0, "negative": 0.0}

        parts = tone_str.split(",")
        try:
            avg_tone = float(parts[0]) if len(parts) > 0 else 0.0
            positive = float(parts[1]) / 10 if len(parts) > 1 else 0.0
            negative = float(parts[2]) / 10 if len(parts) > 2 else 0.0
        except (ValueError, IndexError):
            avg_tone, positive, negative = 0.0, 0.0, 0.0

        return {
            "avg_tone": avg_tone,
            "positive": positive,
            "negative": negative,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = NewsProcessor()

    # 测试
    test_text = """
    <p>据新华社报道，央行宣布降准0.5个百分点，释放长期资金约1万亿元。</p>
    这一举措将有效降低金融机构资金成本，https://example.com 有利于实体经济。
    专家表示，此次降准力度超预期 @财经记者 #货币政策#
    """

    result = processor.process_text(test_text)
    print("清洗后:", result["cleaned"])
    print("分词:", result["tokens"])
    print("去停用词:", result["tokens_filtered"])
