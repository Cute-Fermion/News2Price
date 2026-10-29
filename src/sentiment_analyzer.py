"""
情感分析模块
- FinBERT（HuggingFace 免费模型）
- LLM API（OpenAI 兼容格式，从 .env 读取配置，异步并发，JSON 输出）
- 词典法（baseline，含否定词翻转 + 程度副词加权）

所有词典和参数均从 data/ 文件和 config.yaml 加载，无硬编码常量。
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_llm_config, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)


# ============================================================
# 词典加载工具函数
# ============================================================
def _load_word_list(path_str: str, default_words: set[str] | None = None) -> set[str]:
    """
    从文本文件加载词表（每行一个词，# 开头为注释行）
    支持相对路径（相对于 PROJECT_ROOT/data/）
    """
    if not path_str:
        return default_words or set()

    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        logger.warning(f"词表文件不存在：{p}")
        return default_words or set()

    words = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word)

    logger.info(f"加载词表：{p}，共 {len(words)} 个词")
    return words


def _load_degree_adverbs(path_str: str) -> dict[str, float]:
    """
    从文本文件加载程度副词表（Tab 分隔：词语\t系数）
    """
    if not path_str:
        return {}

    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        logger.warning(f"程度副词表不存在：{p}")
        return {}

    adverbs = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    adverbs[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    continue

    logger.info(f"加载程度副词表：{p}，共 {len(adverbs)} 个")
    return adverbs


# ============================================================
# LLM 分析断点续传工具
# ============================================================

def _make_article_id(date_str: str, title: str) -> str:
    """
    生成稳定的文章唯一 ID（基于 date + title 的 MD5 哈希）
    即使输入顺序变化，同一条新闻的 ID 不变，保证断点续传正确匹配
    """
    raw = f"{str(date_str).strip()}|{str(title).strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path) -> pd.DataFrame | None:
    """加载已有的 checkpoint 文件，返回 DataFrame 或 None"""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, dtype={"_article_id": str})
        logger.info(f"加载断点文件: {path} ({len(df)} 条已完成)")
        return df
    except Exception as e:
        logger.warning(f"加载断点文件失败: {e}")
        return None


def _write_checkpoint_entry(path: Path, article_id: str, date_str: str,
                            score: float, label: str, confidence: float, status: str):
    """追加写入一条 checkpoint 记录（同步写入，asyncio 单线程无竞争）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["_article_id", "date", "llm_score", "llm_label", "llm_confidence", "llm_status"])
        writer.writerow([article_id, date_str, score, label, confidence, status])


# ============================================================
# 词典法情感分析（Baseline + 否定词 + 程度副词）
# ============================================================
class DictionarySentimentAnalyzer:
    """基于词典的情感分析（含否定词翻转 + 程度副词加权）"""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        dict_config = config.get("sentiment", {}).get("dictionary", {})

        # 从文件加载词表
        self.positive_words = _load_word_list(
            dict_config.get("positive_words_path", "positive_words.txt")
        )
        self.negative_words = _load_word_list(
            dict_config.get("negative_words_path", "negative_words.txt")
        )
        self.negation_words = _load_word_list(
            dict_config.get("negation_words_path", "negation_words.txt")
        )
        self.degree_adverbs = _load_degree_adverbs(
            dict_config.get("degree_adverbs_path", "degree_adverbs.txt")
        )

        # 从 config 加载参数
        self.negation_window = dict_config.get("negation_window", 3)
        self.degree_window = dict_config.get("degree_window", 2)
        self.threshold = dict_config.get("threshold", 0.1)

    def analyze(self, tokens: list[str]) -> dict:
        """
        对分词结果进行情感分析，支持：
        1. 否定词翻转：否定词在情感词前 N 个 token 内 → 极性翻转
        2. 程度副词加权：程度副词在情感词前 → 乘以对应系数

        返回：{
            "pos_count": 正面词数（加权后）,
            "neg_count": 负面词数（加权后）,
            "score": 情感得分 (pos-neg)/(pos+neg+1),
            "label": "positive" / "negative" / "neutral",
        }
        """
        pos_score = 0.0
        neg_score = 0.0

        for i, token in enumerate(tokens):
            # 判断当前 token 是否为情感词
            is_positive = token in self.positive_words
            is_negative = token in self.negative_words

            if not is_positive and not is_negative:
                continue

            # 基础权重
            weight = 1.0

            # 1) 检查前方是否有程度副词（取最近的）
            for j in range(max(0, i - self.degree_window), i):
                if tokens[j] in self.degree_adverbs:
                    weight *= self.degree_adverbs[tokens[j]]
                    break  # 只取最近的一个程度副词

            # 2) 检查前方是否有否定词（窗口内）
            negated = False
            for j in range(max(0, i - self.negation_window), i):
                if tokens[j] in self.negation_words:
                    negated = True
                    break

            # 应用权重
            if is_positive:
                if negated:
                    neg_score += weight  # 否定 → 翻转为负面
                else:
                    pos_score += weight
            elif is_negative:
                if negated:
                    pos_score += weight  # 否定负面词 → 翻转为正面
                else:
                    neg_score += weight

        score = (pos_score - neg_score) / (pos_score + neg_score + 1)

        if score > self.threshold:
            label = "positive"
        elif score < -self.threshold:
            label = "negative"
        else:
            label = "neutral"

        return {
            "pos_count": round(pos_score, 2),
            "neg_count": round(neg_score, 2),
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
        self.max_text_length = sent_config.get("max_text_length", 2000)
        self.max_length = sent_config.get("max_length", 512)

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

        logger.info(f"加载 FinBERT 模型：{self.model_name} (device={'cuda' if device >= 0 else 'cpu'})")

        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                truncation=True,
                max_length=self.max_length,
            )
            self._loaded = True
            logger.info("FinBERT 模型加载成功")
        except Exception as e:
            logger.error(f"FinBERT 模型加载失败：{e}")
            raise

    def analyze(self, text: str) -> dict:
        """分析单条文本"""
        self._load_model()

        if not text or not isinstance(text, str):
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        # FinBERT 输出：label (POSITIVE/NEGATIVE/NEUTRAL), score
        try:
            result = self.pipeline(text[:self.max_text_length])[0]  # 截断避免过长
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
            logger.debug(f"FinBERT 分析失败：{e}")
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
                logger.warning(f"FinBERT 批次分析失败：{e}")
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
# LLM API 情感分析（异步并发 + JSON 输出）
# ============================================================
class LLMSentimentAnalyzer:
    """
    使用 LLM API 进行情感分析（批量模式）
    - 每次 API 调用处理 batch_size 篇文章，返回 JSON 数组
    - 使用 AsyncOpenAI + asyncio.Semaphore 实现并发控制
    - 支持任意 OpenAI 兼容接口（DeepSeek / 智谱 GLM / 通义千问 / OpenAI 等）
    配置从 .env 文件读取
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        sent_config = config.get("sentiment", {}).get("llm", {})

        self.llm_config = get_llm_config()
        self.batch_size = sent_config.get("batch_size", 10)
        self.prompt_template = sent_config.get(
            "prompt_template",
            (
                "请分析以下财经新闻的情感倾向和程度。情感倾向包含 positive/negative/neutral，程度取值在 0.0~1.0 范围内，数值越高表示越倾向该情感。"
                "请严格用一下 JSON 格式回复，不要包含其他内容：\n"
                '{{"label": "positive/negative/neutral", "confidence": 0.0~1.0}}\n'
                "新闻内容：{text}"
            ),
        )
        self.batch_prompt_template = sent_config.get(
            "batch_prompt_template",
            (
                "请分析以下 {n} 条财经新闻对沪深300指数的情感倾向。"
                "对每条新闻判断情感（positive/negative/neutral）和置信度（0.0~1.0）。"
                "请严格按 JSON 数组格式回复，不要包含其他内容：\n"
                '[{{"id": 1, "label": "positive/negative/neutral", "confidence": 0.8}}, ...]\n\n'
                "{articles}"
            ),
        )
        self.max_concurrent = sent_config.get("max_concurrent", 5)
        self.timeout = sent_config.get("timeout", 30)
        self.retry_times = sent_config.get("retry_times", 3)
        self.max_text_length = sent_config.get("max_text_length", 1500)
        self.max_tokens = sent_config.get("max_tokens", 300)
        self.temperature = sent_config.get("temperature", 0.0)

        # 每日采样配置
        self.max_news_per_day = sent_config.get("max_news_per_day", 0)  # 0 表示不限制
        self.sampling_strategy = sent_config.get("sampling_strategy", "priority")

        # label 映射（从 config 加载）
        self.positive_labels = set(sent_config.get("positive_labels", ["positive", "pos", "积极", "正面", "利好"]))
        self.negative_labels = set(sent_config.get("negative_labels", ["negative", "neg", "消极", "负面", "利空"]))

        # 兜底置信度（从 config 加载）
        fallback_conf = sent_config.get("fallback_confidence", {})
        self.fallback_confidence = {
            "positive": fallback_conf.get("positive", 0.6),
            "negative": fallback_conf.get("negative", 0.6),
            "neutral": fallback_conf.get("neutral", 0.3),
        }

        self._initialized = False

    def _init_client(self):
        """检查 API 配置是否有效"""
        if self._initialized:
            return True

        base_url = self.llm_config.get("base_url", "")
        api_key = self.llm_config.get("api_key", "")

        if not base_url or not api_key or api_key == "your_api_key_here":
            logger.warning(
                "LLM API 未配置（base_url 或 api_key 为空），"
                "请在 .env 文件中设置 LLM_BASE_URL 和 LLM_API_KEY"
            )
            return False

        self._initialized = True
        return True

    def _create_async_client(self):
        """在异步函数中创建异步客户端（确保事件循环匹配）"""
        from openai import AsyncOpenAI
        import httpx

        base_url = self.llm_config.get("base_url", "")
        api_key = self.llm_config.get("api_key", "")

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self.max_concurrent,
                max_keepalive_connections=min(self.max_concurrent, 200),
            ),
        )
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout,
            http_client=http_client,
        )

    def _parse_json_response(self, text: str) -> dict:
        """
        解析 LLM 返回的 JSON 文本（单条）
        支持：纯 JSON / 包含 ```json ... ``` 的格式 / 混合文本
        """
        text = text.strip()

        # 尝试提取 ```json ... ``` 代码块
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # 尝试提取 { ... } 部分
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            text = text[brace_start:brace_end]

        try:
            data = json.loads(text)
            label = str(data.get("label", "neutral")).strip().lower()
            confidence = float(data.get("confidence", 0.5))

            # 规范化 label
            if label in self.positive_labels:
                label = "positive"
            elif label in self.negative_labels:
                label = "negative"
            else:
                label = "neutral"

            # 规范化 confidence
            confidence = max(0.0, min(1.0, confidence))

            return {"label": label, "confidence": confidence}
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _parse_batch_json_response(self, text: str, expected_ids: list[int]) -> dict[int, dict]:
        """
        解析 LLM 返回的批量 JSON 数组文本
        返回: {id: {"label": ..., "confidence": ...}, ...}
        支持：纯 JSON 数组 / 包含 ```json ... ``` 的格式 / 混合文本
        """
        text = text.strip()

        # 尝试提取 ```json ... ``` 代码块
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # 尝试提取 [ ... ] 部分
        bracket_start = text.find("[")
        bracket_end = text.rfind("]") + 1
        if bracket_start >= 0 and bracket_end > bracket_start:
            text = text[bracket_start:bracket_end]

        try:
            data_list = json.loads(text)
            if not isinstance(data_list, list):
                return {}

            results = {}
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id")
                if item_id is None:
                    continue
                try:
                    item_id = int(item_id)
                except (ValueError, TypeError):
                    continue

                label = str(item.get("label", "neutral")).strip().lower()
                confidence = float(item.get("confidence", 0.5))

                if label in self.positive_labels:
                    label = "positive"
                elif label in self.negative_labels:
                    label = "negative"
                else:
                    label = "neutral"

                confidence = max(0.0, min(1.0, confidence))
                results[item_id] = {"label": label, "confidence": confidence}

            return results
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}

    def _fallback_parse(self, text: str) -> dict:
        """当 JSON 解析失败时，用关键词匹配作为兜底"""
        text = text.strip().lower()
        if any(kw in text for kw in self.positive_labels):
            return {"label": "positive", "confidence": self.fallback_confidence["positive"]}
        elif any(kw in text for kw in self.negative_labels):
            return {"label": "negative", "confidence": self.fallback_confidence["negative"]}
        else:
            return {"label": "neutral", "confidence": self.fallback_confidence["neutral"]}

    def _make_result(self, label: str, confidence: float, status: str) -> dict:
        """构造统一格式的结果"""
        if label == "positive":
            score = confidence
        elif label == "negative":
            score = -confidence
        else:
            score = 0.0
        return {
            "score": round(score, 4),
            "label": label,
            "confidence": round(confidence, 4),
            "status": status,
        }

    async def _analyze_batch_single_call(
        self, client, batch_texts: list[tuple[int, str]], semaphore: asyncio.Semaphore
    ) -> dict[int, dict]:
        """
        单次 API 调用处理一个 batch 的文章
        batch_texts: [(original_index, text), ...]
        返回: {original_index: result_dict, ...}
        """
        # 构造编号后的文章列表
        articles_parts = []
        id_to_idx = {}  # batch_id -> original_index
        for batch_id, (orig_idx, text) in enumerate(batch_texts, start=1):
            truncated = text[:self.max_text_length] if text else "(空)"
            articles_parts.append(f"【第{batch_id}条】{truncated}")
            id_to_idx[batch_id] = orig_idx

        articles_str = "\n".join(articles_parts)
        prompt = self.batch_prompt_template.format(n=len(batch_texts), articles=articles_str)
        model = self.llm_config.get("model_name", "gpt-3.5-turbo")
        expected_ids = list(id_to_idx.keys())

        # 批量调用需要更多 token
        batch_max_tokens = max(self.max_tokens, len(batch_texts) * 50)

        async with semaphore:
            for attempt in range(1, self.retry_times + 1):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=batch_max_tokens,
                    )
                    result_text = response.choices[0].message.content.strip()

                    # 尝试解析 JSON 数组
                    parsed_batch = self._parse_batch_json_response(result_text, expected_ids)

                    # 为每个文章生成结果
                    batch_results = {}
                    for batch_id, orig_idx in id_to_idx.items():
                        if batch_id in parsed_batch:
                            item = parsed_batch[batch_id]
                            batch_results[orig_idx] = self._make_result(
                                item["label"], item["confidence"], "success"
                            )
                        else:
                            # 该文章在 batch 响应中缺失，用关键词兜底
                            text = dict(batch_texts).get(orig_idx, "")
                            fallback = self._fallback_parse(text)
                            batch_results[orig_idx] = self._make_result(
                                fallback["label"], fallback["confidence"], "fallback_keyword"
                            )

                    return batch_results

                except Exception as e:
                    logger.debug(f"LLM 批量分析失败 (第{attempt}次): {e}")
                    if attempt < self.retry_times:
                        await asyncio.sleep(1)

        # 所有重试都失败，对每篇文章用关键词兜底
        batch_results = {}
        for batch_id, (orig_idx, text) in enumerate(batch_texts, start=1):
            fallback = self._fallback_parse(text)
            batch_results[orig_idx] = self._make_result(
                fallback["label"], fallback["confidence"], "failed"
            )
        return batch_results

    async def _analyze_all_async(
        self,
        texts: list[str],
        article_ids: list[str] | None = None,
        dates: list[str] | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """
        异步批量分析所有文本（分批 + 并发）
        将 texts 分成多个 batch，每个 batch 一次 API 调用，多个 batch 并发执行
        如果提供 article_ids / dates / checkpoint_path，每完成一个 batch 立即追加写入临时文件
        """
        client = self._create_async_client()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        total = len(texts)

        # 构造 (index, text) 列表，过滤空文本
        indexed_texts = []
        empty_indices = set()
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str) or not text.strip():
                empty_indices.add(i)
            else:
                indexed_texts.append((i, text))

        # 分成 batches
        batches = []
        for start in range(0, len(indexed_texts), self.batch_size):
            batch = indexed_texts[start : start + self.batch_size]
            batches.append(batch)

        num_batches = len(batches)
        logger.info(
            f"LLM 批量分析：{total} 条文本，{len(indexed_texts)} 条有效，"
            f"{num_batches} 个批次 (batch_size={self.batch_size}, "
            f"并发={self.max_concurrent})"
        )

        completed_count = [0]

        async def _process_batch(batch):
            result = await self._analyze_batch_single_call(client, batch, semaphore)
            completed_count[0] += 1
            if completed_count[0] % 50 == 0 or completed_count[0] == num_batches:
                logger.info(f"LLM 批次进度：{completed_count[0]}/{num_batches}")
            # 断点续传：每完成一个 batch 立即写入临时文件
            if checkpoint_path and article_ids:
                for batch_id, (orig_idx, _) in enumerate(batch, start=1):
                    if orig_idx in result:
                        r = result[orig_idx]
                        aid = article_ids[orig_idx]
                        d = dates[orig_idx] if dates else ""
                        _write_checkpoint_entry(
                            checkpoint_path, aid, d,
                            r["score"], r["label"], r["confidence"], r["status"],
                        )
            return result

        try:
            batch_results_list = await asyncio.gather(
                *[_process_batch(batch) for batch in batches]
            )
        finally:
            await client.close()

        # 合并所有 batch 结果
        all_results = {}
        for batch_result in batch_results_list:
            all_results.update(batch_result)

        # 构造最终结果列表（保持输入顺序）
        final_results = []
        for i in range(total):
            if i in empty_indices:
                final_results.append(self._make_result("neutral", 0.0, "empty"))
            elif i in all_results:
                final_results.append(all_results[i])
            else:
                # 不应该发生，但作为安全兜底
                final_results.append(self._make_result("neutral", 0.0, "failed"))

        return final_results

    def analyze(self, text: str) -> dict:
        """分析单条文本（同步接口，内部使用 asyncio）"""
        if not self._init_client():
            return {"score": 0.0, "label": "neutral", "confidence": 0.0, "status": "failed"}

        return asyncio.run(self._analyze_all_async([text]))[0]

    def _sample_daily(self, df: pd.DataFrame, text_col: str) -> tuple[pd.DataFrame, list[bool]]:
        """
        每日采样，返回 (采样后的 df, 是否被采样的 mask)
        mask[i] = True 表示第 i 条被选中参与 LLM 分析
        """
        if self.max_news_per_day <= 0:
            # 不限制
            return df, [True] * len(df)

        if "date" not in df.columns:
            logger.warning("DataFrame 无 date 列，跳过每日采样")
            return df, [True] * len(df)

        df = df.copy()
        df["_orig_idx"] = df.index  # 保留原始索引
        sampled_mask = [False] * len(df)

        # 按日期分组采样
        for date, group in df.groupby("date"):
            n = len(group)
            if n <= self.max_news_per_day:
                # 全部选中
                for idx in group["_orig_idx"]:
                    sampled_mask[idx] = True
                continue

            # 需要采样
            if self.sampling_strategy == "priority":
                # 优先选有正文的（text_col 非空且长度 > 10）
                has_content = group[text_col].fillna("").str.len() > 10
                with_content = group[has_content]
                without_content = group[~has_content]

                # 先选有正文的（最多 max_news_per_day 条）
                selected = []
                if len(with_content) >= self.max_news_per_day:
                    # 有正文的已经够了，随机选
                    selected = with_content.sample(n=self.max_news_per_day, random_state=42)
                else:
                    # 全选有正文的，剩余名额从无正文的中随机选
                    selected = with_content
                    remaining = self.max_news_per_day - len(with_content)
                    if remaining > 0 and len(without_content) > 0:
                        extra = without_content.sample(
                            n=min(remaining, len(without_content)),
                            random_state=42,
                        )
                        selected = pd.concat([selected, extra])
            else:
                # 纯随机
                selected = group.sample(n=self.max_news_per_day, random_state=42)

            for idx in selected["_orig_idx"]:
                sampled_mask[idx] = True

        sampled_count = sum(sampled_mask)
        logger.info(
            f"每日采样：{len(df)} 条 -> {sampled_count} 条 "
            f"(max_news_per_day={self.max_news_per_day}, strategy={self.sampling_strategy})"
        )
        return df, sampled_mask

    def analyze_dataframe(
        self, df: pd.DataFrame, text_col: str = "cleaned_text", force: bool = False
    ) -> pd.DataFrame:
        """
        批量分析 DataFrame（异步并发 + 断点续传）
        使用 asyncio + Semaphore 控制并发数
        每完成一个 batch 立即写入 checkpoint 文件，中断后可恢复
        force=True 时忽略已有 checkpoint，全部重新分析（旧 checkpoint 保留不删除）
        返回结果包含 llm_status 列用于区分：
        - success: LLM 成功返回并解析 JSON
        - fallback_keyword: LLM 返回但 JSON 解析失败，使用关键词兜底
        - failed: API 调用失败（重试耗尽）
        - empty: 输入文本为空
        - sampled_out: 每日采样被剔除（未调用 LLM）
        - checkpoint: 从断点文件恢复的历史结果
        """
        if not self._init_client():
            logger.warning("LLM 客户端未初始化，跳过 LLM 情感分析")
            df = df.copy()
            df["llm_score"] = 0.0
            df["llm_label"] = "neutral"
            df["llm_confidence"] = 0.0
            df["llm_status"] = "failed"
            return df

        df = df.copy()

        # ---- 断点续传：准备 ----
        # force=True 时使用新文件名，避免旧记录覆盖新结果；旧文件保留不删除
        if force:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = PROJECT_ROOT / "data" / "llm_temp" / f"llm_checkpoint_{ts}.csv"
            logger.info(f"--force 模式：使用新断点文件 {checkpoint_path}")
        else:
            checkpoint_path = PROJECT_ROOT / "data" / "llm_temp" / "llm_checkpoint.csv"
        checkpoint_df = None if force else _load_checkpoint(checkpoint_path)
        checkpoint_map = {}  # article_id -> (score, label, confidence, status)
        if checkpoint_df is not None and not checkpoint_df.empty:
            for _, row in checkpoint_df.iterrows():
                aid = str(row["_article_id"])
                if aid not in checkpoint_map:  # 保留首条（避免重复）
                    checkpoint_map[aid] = (
                        float(row["llm_score"]),
                        str(row["llm_label"]),
                        float(row["llm_confidence"]),
                        str(row["llm_status"]),
                    )
            logger.info(f"断点文件已有 {len(checkpoint_map)} 条唯一记录")

        # ---- 生成稳定 ID 和日期字符串 ----
        date_col = df["date"] if "date" in df.columns else pd.Series([""] * len(df))
        title_col = df["title"] if "title" in df.columns else pd.Series([""] * len(df))
        article_ids = [
            _make_article_id(str(d)[:10], str(t))
            for d, t in zip(date_col, title_col)
        ]
        date_strs = [str(d)[:10] for d in date_col]

        # ---- 每日采样 ----
        df, sampled_mask = self._sample_daily(df, text_col)

        # ---- 过滤已处理 + 采样剔除的，只对待处理文章调用 LLM ----
        texts_all = df[text_col].fillna("").astype(str).tolist()
        pending_indices = []  # 需要调用 LLM 的行号
        for i, (text, sampled) in enumerate(zip(texts_all, sampled_mask)):
            if not sampled:
                continue
            if article_ids[i] in checkpoint_map:
                continue  # checkpoint 已有
            if not text.strip():
                continue  # 空文本，不需要 LLM
            pending_indices.append(i)

        already_count = sum(1 for i in range(len(df)) if sampled_mask[i] and article_ids[i] in checkpoint_map)
        if already_count > 0:
            logger.info(f"断点续传：跳过已有 {already_count} 条，待处理 {len(pending_indices)} 条")

        # ---- 调用 LLM ----
        if pending_indices:
            pending_texts = [texts_all[i] for i in pending_indices]
            pending_aids = [article_ids[i] for i in pending_indices]
            pending_dates = [date_strs[i] for i in pending_indices]

            logger.info(
                f"LLM 异步分析 {len(pending_texts)} 条文本 "
                f"(batch_size={self.batch_size}, max_concurrent={self.max_concurrent})..."
            )

            new_results = asyncio.run(self._analyze_all_async(
                pending_texts,
                article_ids=pending_aids,
                dates=pending_dates,
                checkpoint_path=checkpoint_path,
            ))
            # 合并到 checkpoint_map（文件写入已在 _analyze_all_async 中逐 batch 完成）
            for idx_in_pending, orig_i in enumerate(pending_indices):
                r = new_results[idx_in_pending]
                checkpoint_map[pending_aids[idx_in_pending]] = (
                    r["score"], r["label"], r["confidence"], r["status"],
                )
        else:
            logger.info("无需调用 LLM（全部已有 checkpoint 或为空文本）")

        # ---- 组装最终结果 ----
        llm_scores = []
        llm_labels = []
        llm_confs = []
        llm_statuses = []

        for i in range(len(df)):
            if not sampled_mask[i]:
                # 采样剔除
                llm_scores.append(0.0)
                llm_labels.append("neutral")
                llm_confs.append(0.0)
                llm_statuses.append("sampled_out")
            elif article_ids[i] in checkpoint_map:
                # 从 checkpoint 读取（含本次新结果 + 历史断点结果）
                s, l, c, st = checkpoint_map[article_ids[i]]
                llm_scores.append(s)
                llm_labels.append(l)
                llm_confs.append(c)
                llm_statuses.append(st)
            else:
                # 空文本等未走 LLM 的
                llm_scores.append(0.0)
                llm_labels.append("neutral")
                llm_confs.append(0.0)
                llm_statuses.append("empty")

        df["llm_score"] = llm_scores
        df["llm_label"] = llm_labels
        df["llm_confidence"] = llm_confs
        df["llm_status"] = llm_statuses

        # 统计并输出各状态数量
        status_counts = df["llm_status"].value_counts()
        total = len(df)
        logger.info("=" * 50)
        logger.info("LLM 情感分析状态统计：")
        for status in ["success", "fallback_keyword", "failed", "empty", "sampled_out"]:
            count = status_counts.get(status, 0)
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {status}: {count} ({pct:.1f}%)")
        logger.info(f"  checkpoint 文件: {checkpoint_path}")
        logger.info("=" * 50)

        return df


# ============================================================
# 统一情感分析入口
# ============================================================
def run_sentiment_analysis(
    df: pd.DataFrame,
    config: dict | None = None,
    method: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    运行情感分析
    method: "dictionary" / "finbert" / "llm" / "both"
    force: True 时忽略 LM 断点文件，全部重新分析
    """
    if config is None:
        config = load_config()
    if method is None:
        method = config.get("sentiment", {}).get("method", "both")

    df = df.copy()

    # 1. 词典法
    if method in ("dictionary", "both"):
        logger.info("执行词典法情感分析...")
        dict_analyzer = DictionarySentimentAnalyzer(config)
        df = dict_analyzer.analyze_dataframe(df)

    # 2. FinBERT
    if method in ("finbert", "both"):
        logger.info("执行 FinBERT 情感分析...")
        try:
            finbert = FinBERTAnalyzer(config)
            df = finbert.analyze_dataframe(df)
        except Exception as e:
            logger.error(f"FinBERT 分析失败：{e}")
            df["finbert_score"] = 0.0
            df["finbert_label"] = "neutral"
            df["finbert_confidence"] = 0.0

    # 3. LLM API
    if method in ("llm", "both"):
        logger.info("执行 LLM 情感分析...")
        try:
            llm = LLMSentimentAnalyzer(config)
            df = llm.analyze_dataframe(df, force=force)
        except Exception as e:
            logger.error(f"LLM 分析失败：{e}")
            df["llm_score"] = 0.0
            df["llm_label"] = "neutral"
            df["llm_confidence"] = 0.0

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 测试词典法
    config = load_config()
    analyzer = DictionarySentimentAnalyzer(config)
    test_tokens = ["央行", "宣布", "降准", "释放", "资金", "利好", "股市", "上涨"]
    result = analyzer.analyze(test_tokens)
    print("词典法结果:", result)
