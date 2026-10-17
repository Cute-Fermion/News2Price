"""
情感分析模块
- FinBERT（HuggingFace 免费模型）
- LLM API（OpenAI 兼容格式，从 .env 读取配置）
- 词典法（baseline，使用 Loughran-McDonald 中文映射）
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_llm_config, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ============================================================
# 词典法情感分析（Baseline）
# ============================================================
# 简化版金融情感词典（正面/负面）
# 实际项目中可扩展为完整的 Loughran-McDonald 中文映射版
_POSITIVE_WORDS = set(
    """
    上涨 增长 盈利 利润 收益 利好 突破 创新 新高 牛市 涨停 升值 扩张
    超预期 强劲 繁荣 回升 反弹 复苏 改善 优化 提升 积极 乐观 看好
    买入 增持 推荐 跑赢 领先 突破 放量 资金流入 主力 机构看好
    降准 降息 宽松 刺激 扶持 补贴 减免 优惠 支持 鼓励
    """.split()
)

_NEGATIVE_WORDS = set(
    """
    下跌 下降 亏损 亏损 利空 暴跌 熊市 跌停 贬值 收缩 衰退 恶化
    低于预期 疲软 低迷 下滑 回落 调整 风险 危机 恐慌 悲观 看空
    卖出 减持 评级下调 跑输 落后 缩量 资金流出 套牢 爆仓 违约
    加息 收紧 监管 处罚 调查 诉讼 退市 警告 警示 停牌
    通胀 紧缩 债务 破产 裁员 亏损 赤字 泡沫 崩盘 熔断
    """.split()
)


class DictionarySentimentAnalyzer:
    """基于词典的情感分析（baseline）"""

    def __init__(self):
        self.positive_words = _POSITIVE_WORDS
        self.negative_words = _NEGATIVE_WORDS

    def analyze(self, tokens: list[str]) -> dict:
        """
        对分词结果进行情感分析
        返回: {
            "pos_count": 正面词数,
            "neg_count": 负面词数,
            "score": 情感得分 (pos-neg)/(pos+neg+1),
            "label": "positive" / "negative" / "neutral",
        }
        """
        pos_count = sum(1 for t in tokens if t in self.positive_words)
        neg_count = sum(1 for t in tokens if t in self.negative_words)

        score = (pos_count - neg_count) / (pos_count + neg_count + 1)

        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "pos_count": pos_count,
            "neg_count": neg_count,
            "score": round(score, 4),
            "label": label,
        }

    def analyze_dataframe(
        self, df: pd.DataFrame, token_col: str = "tokens_filtered"
    ) -> pd.DataFrame:
        """批量分析 DataFrame"""
        df = df.copy()
        results = df[token_col].apply(
            lambda tokens: self.analyze(tokens if isinstance(tokens, list) else [])
        )
        df["dict_pos_count"] = results.apply(lambda x: x["pos_count"])
        df["dict_neg_count"] = results.apply(lambda x: x["neg_count"])
        df["dict_score"] = results.apply(lambda x: x["score"])
        df["dict_label"] = results.apply(lambda x: x["label"])
        return df


# ============================================================
# FinBERT 情感分析
# ============================================================
class FinBERTAnalyzer:
    """
    使用 FinBERT 模型进行情感分析
    默认使用 yiyanghkust/finbert-tone（免费，HuggingFace）
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        sent_config = config.get("sentiment", {}).get("finbert", {})

        self.model_name = sent_config.get("model_name", "yiyanghkust/finbert-tone")
        self.batch_size = sent_config.get("batch_size", 32)
        self.device = sent_config.get("device", "auto")

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._loaded = False

    def _load_model(self):
        """延迟加载模型"""
        if self._loaded:
            return

        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

        if self.device == "auto":
            import torch
            device = 0 if torch.cuda.is_available() else -1
        else:
            device = 0 if self.device == "cuda" else -1

        logger.info(f"加载 FinBERT 模型: {self.model_name} (device={'cuda' if device >= 0 else 'cpu'})")

        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                truncation=True,
                max_length=512,
            )
            self._loaded = True
            logger.info("FinBERT 模型加载成功")
        except Exception as e:
            logger.error(f"FinBERT 模型加载失败: {e}")
            raise

    def analyze(self, text: str) -> dict:
        """分析单条文本"""
        self._load_model()

        if not text or not isinstance(text, str):
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        # FinBERT 输出: label (POSITIVE/NEGATIVE/NEUTRAL), score
        try:
            result = self.pipeline(text[:2000])[0]  # 截断避免过长
            label = result["label"].lower()
            score = result["score"]

            # 统一分数到 [-1, 1]
            if label == "positive":
                sentiment_score = score
            elif label == "negative":
                sentiment_score = -score
            else:
                sentiment_score = 0.0

            return {
                "score": round(sentiment_score, 4),
                "label": label,
                "confidence": round(score, 4),
            }
        except Exception as e:
            logger.debug(f"FinBERT 分析失败: {e}")
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """批量分析"""
        self._load_model()

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # 过滤空文本
            valid_batch = [t if t and isinstance(t, str) else " " for t in batch]

            try:
                batch_results = self.pipeline(valid_batch)
                for r in batch_results:
                    label = r["label"].lower()
                    score = r["score"]
                    if label == "positive":
                        sentiment_score = score
                    elif label == "negative":
                        sentiment_score = -score
                    else:
                        sentiment_score = 0.0
                    results.append({
                        "score": round(sentiment_score, 4),
                        "label": label,
                        "confidence": round(score, 4),
                    })
            except Exception as e:
                logger.warning(f"FinBERT 批次分析失败: {e}")
                results.extend([{"score": 0.0, "label": "neutral", "confidence": 0.0}] * len(batch))

        return results

    def analyze_dataframe(
        self, df: pd.DataFrame, text_col: str = "cleaned_text"
    ) -> pd.DataFrame:
        """批量分析 DataFrame"""
        df = df.copy()
        texts = df[text_col].fillna("").astype(str).tolist()

        logger.info(f"FinBERT 分析 {len(texts)} 条文本...")
        results = self.analyze_batch(texts)

        df["finbert_score"] = [r["score"] for r in results]
        df["finbert_label"] = [r["label"] for r in results]
        df["finbert_confidence"] = [r["confidence"] for r in results]

        return df


# ============================================================
# LLM API 情感分析
# ============================================================
class LLMSentimentAnalyzer:
    """
    使用 LLM API 进行情感分析
    支持任意 OpenAI 兼容接口（DeepSeek / 智谱GLM / 通义千问 / OpenAI 等）
    配置从 .env 文件读取
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        sent_config = config.get("sentiment", {}).get("llm", {})

        self.llm_config = get_llm_config()
        self.prompt_template = sent_config.get(
            "prompt_template",
            "请分析以下财经新闻的情感倾向，只回复一个词：positive、negative 或 neutral。\n新闻内容：{text}",
        )
        self.max_concurrent = sent_config.get("max_concurrent", 5)
        self.timeout = sent_config.get("timeout", 30)
        self.retry_times = sent_config.get("retry_times", 3)

        self.client = None
        self._initialized = False

    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        if self._initialized:
            return

        base_url = self.llm_config.get("base_url", "")
        api_key = self.llm_config.get("api_key", "")

        if not base_url or not api_key or api_key == "your_api_key_here":
            logger.warning(
                "LLM API 未配置（base_url 或 api_key 为空），"
                "请在 .env 文件中设置 LLM_BASE_URL 和 LLM_API_KEY"
            )
            return

        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
        self._initialized = True
        logger.info(f"LLM 客户端初始化成功: {base_url}")

    def analyze(self, text: str) -> dict:
        """分析单条文本"""
        self._init_client()

        if self.client is None:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        if not text or not isinstance(text, str):
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        prompt = self.prompt_template.format(text=text[:1500])  # 截断避免过长
        model = self.llm_config.get("model_name", "gpt-3.5-turbo")

        for attempt in range(1, self.retry_times + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # 确定性输出
                    max_tokens=10,
                )
                result_text = response.choices[0].message.content.strip().lower()

                # 解析结果
                if "positive" in result_text:
                    return {"score": 0.8, "label": "positive", "confidence": 0.8}
                elif "negative" in result_text:
                    return {"score": -0.8, "label": "negative", "confidence": 0.8}
                else:
                    return {"score": 0.0, "label": "neutral", "confidence": 0.5}

            except Exception as e:
                logger.debug(f"LLM 分析失败 (第{attempt}次): {e}")
                if attempt < self.retry_times:
                    time.sleep(1)

        return {"score": 0.0, "label": "neutral", "confidence": 0.0}

    def analyze_dataframe(
        self, df: pd.DataFrame, text_col: str = "cleaned_text"
    ) -> pd.DataFrame:
        """
        批量分析 DataFrame
        注意：LLM API 有速率限制，逐条调用
        """
        self._init_client()

        if self.client is None:
            logger.warning("LLM 客户端未初始化，跳过 LLM 情感分析")
            df = df.copy()
            df["llm_score"] = 0.0
            df["llm_label"] = "neutral"
            df["llm_confidence"] = 0.0
            return df

        df = df.copy()
        texts = df[text_col].fillna("").astype(str).tolist()

        logger.info(f"LLM 分析 {len(texts)} 条文本...")
        scores, labels, confidences = [], [], []

        for i, text in enumerate(texts):
            result = self.analyze(text)
            scores.append(result["score"])
            labels.append(result["label"])
            confidences.append(result["confidence"])

            # 进度日志
            if (i + 1) % 100 == 0:
                logger.info(f"LLM 分析进度: {i + 1}/{len(texts)}")

            # 避免速率限制
            time.sleep(0.2)

        df["llm_score"] = scores
        df["llm_label"] = labels
        df["llm_confidence"] = confidences

        return df


# ============================================================
# 统一情感分析入口
# ============================================================
def run_sentiment_analysis(
    df: pd.DataFrame,
    config: dict | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """
    运行情感分析
    method: "dictionary" / "finbert" / "llm" / "both"
    """
    if config is None:
        config = load_config()
    if method is None:
        method = config.get("sentiment", {}).get("method", "both")

    df = df.copy()

    # 1. 词典法（总是执行，作为 baseline）
    logger.info("执行词典法情感分析...")
    dict_analyzer = DictionarySentimentAnalyzer()
    df = dict_analyzer.analyze_dataframe(df)

    # 2. FinBERT
    if method in ("finbert", "both"):
        logger.info("执行 FinBERT 情感分析...")
        try:
            finbert = FinBERTAnalyzer(config)
            df = finbert.analyze_dataframe(df)
        except Exception as e:
            logger.error(f"FinBERT 分析失败: {e}")
            df["finbert_score"] = 0.0
            df["finbert_label"] = "neutral"
            df["finbert_confidence"] = 0.0

    # 3. LLM API
    if method in ("llm", "both"):
        logger.info("执行 LLM 情感分析...")
        try:
            llm = LLMSentimentAnalyzer(config)
            df = llm.analyze_dataframe(df)
        except Exception as e:
            logger.error(f"LLM 分析失败: {e}")
            df["llm_score"] = 0.0
            df["llm_label"] = "neutral"
            df["llm_confidence"] = 0.0

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 测试词典法
    analyzer = DictionarySentimentAnalyzer()
    test_tokens = ["央行", "宣布", "降准", "释放", "资金", "利好", "股市", "上涨"]
    result = analyzer.analyze(test_tokens)
    print("词典法结果:", result)
