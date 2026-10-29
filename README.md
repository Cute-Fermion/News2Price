# News2Price — 基于新闻文本的A股股价波动预测与因果关系研究

> 探索中文财经新闻情绪与沪深300指数价格波动之间的关系，检验新闻情绪是否具有市场"预测能力"。

## 项目概述

本项目以 Tetlock (2007) 的经典研究"Giving Content to Investor Sentiment"为理论基石，结合 Kirtac & Germano (2024) 关于大语言模型在金融情感分析中的前沿成果，针对中国A股市场构建一套完整的"新闻情绪→股价预测"研究框架。

核心问题：**中文财经新闻的情绪指标能否预测A股指数的未来收益与波动？**

### 研究结论（基于 Run0020，词典法，2022-2026）

- 新闻情感整体偏正面（均值 0.144），中文财经新闻存在"报喜不报忧"的系统性偏差
- 情感与收益的相关性极弱（r=0.06），Granger 因果检验不显著（p=0.28），**词典法情感不能作为收益的有效预测因子**
- 情感能显著预测交易量变化（Granger p=0.001，回归 R²=10.3%），说明情感更多影响交易行为而非价格方向
- 存在反向因果：收益显著 Granger-cause 情感（p=0.028），新闻情感很大程度上是对已发生行情的反映

详细分析报告见 `outputs/runs/run0020_20261028_210220/analysis_report.md`。

---

## 理论基础

### Tetlock (2007) 核心发现

Tetlock 使用 General Inquirer 对《华尔街日报》"Abreast of the Market"专栏进行量化内容分析，通过主成分分析从77个词汇类别中提取"悲观因子"（Pessimism Factor），然后用向量自回归（VAR）模型检验其与道琼斯指数收益的关系：

- **高媒体悲观度 → 预测次日市场下跌**（平均影响 -8.1 bps，大于无条件均值 5.4 bps）
- **下跌在一周内完全反转** → 支持"噪声交易者/流动性交易"理论，而非"新信息"理论
- **极端悲观或乐观 → 预测高交易量**
- **低市场收益 → 预测次日更高媒体悲观度**（正反馈交易）

### Kirtac & Germano (2024) 核心发现

对比 OPT、BERT、FinBERT 和 Loughran-McDonald 词典在 96.5 万篇美国财经新闻上的表现：

| 模型          | 准确率   | 多空策略收益 | 夏普比率 |
| ----------- | ----- | ------ | ---- |
| OPT (GPT-3) | 74.4% | 355%   | 3.05 |
| BERT        | 72.5% | 235%   | —    |
| FinBERT     | 72.2% | 165%   | —    |
| L-M 词典      | 50.1% | 0.91%  | 1.23 |

**关键启示**：LLM 方法显著优于传统词典法，但词典法作为 baseline 仍有价值。

### A股适配考量

将上述框架迁移到A股需注意以下市场特性：

- **T+1 交易制度**：当日买入次日才能卖出，影响事件窗口设计
- **涨跌停板**（±10%/±20%）：极端情绪日可能无法充分反映价格
- **散户占比高**：噪声交易者效应可能比美股更显著
- **政策驱动性强**：政策类新闻的影响可能远大于公司层面新闻
- **中文NLP特殊性**：需要中文分词、中文情感词典或中文预训练模型

---

## 项目结构

```
News2Price/
├── README.md                      # 本文件
├── .env                           # 敏感配置（Tushare token、LLM API key）— 不提交 git
├── .gitignore
├── config.yaml                    # 所有可调参数（标的、日期、模型参数、VAR配置等）
├── requirements.txt               # Python 依赖
├── main.py                        # 命令行入口
│
├── src/                           # 核心源代码
│   ├── __init__.py
│   ├── utils.py                   # 配置加载、运行管理、日志系统
│   ├── data_collector.py          # Tushare/AkShare/GDELT 多源数据采集
│   ├── news_processor.py          # 中文分词(jieba)、文本清洗、GDELT解析
│   ├── news_filter.py             # 黑名单/白名单关键词过滤
│   ├── sentiment_analyzer.py      # 词典法 + FinBERT + LLM API 三轨情感分析
│   ├── feature_engineer.py        # 日度情感指数、衰减加权、极端标记、哑变量构造
│   ├── analysis.py                # VAR模型、Granger检验、OLS回归、脉冲响应
│   └── visualization.py           # 行情图、情感图、IRF、仪表板
│
├── scripts/                       # 辅助脚本
│   ├── filter_news.py             # 新闻过滤（独立运行，支持 --force）
│   ├── fetch_tushare_news.py      # Tushare 新闻批量拉取
│   └── test_llm_*.py              # LLM API 测试脚本
│
├── words/                         # 词典与词表
│   ├── positive_words.txt         # 正面情感词（44个）
│   ├── negative_words.txt         # 负面情感词（51个）
│   ├── negation_words.txt         # 否定词（23个）
│   ├── degree_adverbs.txt         # 程度副词（28个，Tab分隔带系数）
│   ├── stopwords.txt              # 中文停用词（478个）
│   ├── blacklist.txt              # 新闻过滤黑名单
│   └── whitelist.txt              # 新闻过滤白名单
│
├── data/
│   ├── tushare_news/              # Tushare 原始新闻缓存
│   │   ├── news_sina.csv          # 新浪财经快讯（885K条，合并自两个目录）
│   │   ├── major_news.csv         # 新闻通讯（582K条）
│   │   └── cctv_news.csv          # 央视新闻联播（24K条）
│   ├── filtered_news/             # 过滤后新闻
│   │   └── filtered_news.csv      # 黑名单+白名单过滤结果（238K条）
│   ├── raw/                       # 原始数据（自动保存）
│   │   ├── stock_data.csv
│   │   └── news_data.csv
│   ├── processed/                 # 处理后数据
│   │   └── news_sentiment_*.csv   # 各方法情感分析结果
│   └── llm_temp/                  # LLM 断点续传临时文件
│       └── llm_checkpoint.csv     # 断点文件（不自动删除）
│
├── outputs/
│   └── runs/                      # 运行输出（自动递增编号）
│       └── run00XX_YYYYMMDD_HHMMSS/
│           ├── config.json        # 本次运行的配置快照
│           ├── run.log            # 运行日志
│           ├── figures/           # 图表（6张）
│           │   ├── stock_price.png
│           │   ├── sentiment_avg_sentiment.png
│           │   ├── sentiment_vs_return_lag1_avg_sentiment.png
│           │   ├── sentiment_distribution_avg_sentiment.png
│           │   ├── dashboard.png
│           │   └── impulse_response.png
│           ├── processed/
│           │   └── merged_dataset.csv   # 最终合并数据集
│           └── analysis_report.md       # 分析报告（手动生成）
│
├── notebooks/
│   └── main_analysis.ipynb        # 端到端分析 Notebook
└── papers/                        # 参考文献
    ├── Tetlock-GivingContentInvestor-2007.pdf
    ── 2412.19245v1.pdf
```

---

## 数据方案

### 股票数据

| 数据源 | 接口 | 内容 | 积分要求 |
|--------|------|------|---------|
| **Tushare** | `index_daily` | 沪深300日频行情（OHLCV） | 基础接口，2000积分可用 |
| **AkShare** | 备用 | 行情补充 | 免费 |

### 新闻数据

| 数据源 | 说明 | 规模 | 优先级 |
|--------|------|------|--------|
| **Tushare** | `news`(新浪快讯)、`major_news`(新闻通讯)、`cctv_news`(新闻联播) | 1.5M+ 条 | 主力 |
| **AkShare** | 东方财富/新浪财经财经新闻 | 补充 | 备用 |
| **GDELT** | 全球新闻事件数据库 | 覆盖广 | 补充 |

### 新闻过滤

通过 `scripts/filter_news.py` 对原始新闻进行黑名单/白名单关键词过滤：
- 黑名单：排除广告、娱乐等无关内容
- 白名单：至少命中 2 个财经关键词才保留
- 过滤后保留率约 17.6%（1.35M → 238K 条）

---

## 方法论

### 完整处理流水线

```
数据采集 → 新闻过滤 → 文本处理 → 情感分析 → 特征工程 → 统计分析 → 可视化
```

### Phase 1: 数据采集与预处理

1. 选取目标标的（沪深300，代码 `399300.SZ`）
2. 获取日频行情数据，计算日收益率 `return_t = (close_t - close_{t-1}) / close_{t-1}`
3. 从 Tushare 拉取多源新闻数据，按日期聚合
4. 通过 `scripts/filter_news.py` 进行黑名单/白名单过滤

### Phase 2: 文本处理与情感分析

#### 文本处理（`src/news_processor.py`）

- 中文分词：jieba
- 文本清洗：去除特殊字符、HTML标签、多余空白
- 停用词过滤：478 个中文停用词
- 空 content 自动回退到 title 列

#### 情感分析三轨制（`src/sentiment_analyzer.py`）

**1. 词典法（Dictionary-based）— Baseline**

- 正面词表（44词）+ 负面词表（51词）
- 否定词翻转：否定词在情感词前 N 个 token 内 → 极性翻转
- 程度副词加权：程度副词在情感词前 → 乘以对应系数
- 情感得分：`score = (pos - neg) / (pos + neg + 1)`

**2. FinBERT**

- 模型：`yiyanghkust/finbert-tone`（HuggingFace）
- 批量推理，支持 GPU 加速

**3. LLM API（异步批量 + 断点续传）**

- 支持任意 OpenAI 兼容接口（DeepSeek / 智谱 GLM / 通义千问 / OpenAI）
- 批量模式：每次 API 调用处理 10 篇文章，返回 JSON 数组
- 异步并发：`asyncio + Semaphore` 控制并发数（默认 20）
- 每日采样：`max_news_per_day` 限制每日处理量，控制 API 成本
- **断点续传**：每完成一个 batch 立即写入 `data/llm_temp/llm_checkpoint.csv`，中断后自动恢复
- 状态追踪：`success` / `fallback_keyword` / `failed` / `empty` / `sampled_out`

### Phase 3: 特征工程（`src/feature_engineer.py`）

| 指标 | 定义 | 说明 |
|------|------|------|
| `avg_sentiment` | 当日平均情感得分 | 核心变量 |
| `sentiment_std` | 当日情感标准差 | 意见分歧度 |
| `news_count` | 当日新闻条数 | 新闻强度/关注度 |
| `pos_ratio` / `neg_ratio` | 正/负面新闻比例 | 情感结构 |
| `avg_sentiment_decay` | 指数衰减加权情感 | 处理非交易日新闻 |
| `extreme_sentiment` | 极端情绪标记（±1） | 滚动30日窗口，避免 look-ahead bias |
| `volume_detrend` | 去趋势交易量（对数-MA20） | 消除长期趋势 |
| 星期/月份哑变量 | 5个星期 + 12个月份 | 控制日历效应 |

### Phase 4: 统计分析（`src/analysis.py`）

#### VAR 模型

```
内生变量: return, avg_sentiment, volume_detrend
外生变量: 星期哑变量 + 月份哑变量
滞后阶数: 5（AIC/BIC 自动选择）
```

#### Granger 因果检验

检验新闻情绪是否 Granger-cause 股票收益，以及反向关系。

#### OLS 回归

- 基准回归：`return_t = α + β × sentiment_{t-1} + controls + ε`
- 反向因果检验：`sentiment_t = α + β × return_{t-1} + controls + ε`
- 交易量回归：`volume_t = α + γ × |extreme_sentiment| + controls + ε`

#### 脉冲响应分析（IRF）

正交化脉冲响应函数，展示三变量系统的动态交互。

### Phase 5: 可视化（`src/visualization.py`）

自动生成 6 张图表：行情走势、情感指数、情感-收益散点图、情感分布、综合仪表盘、脉冲响应。

---

## 技术栈

| 类别 | 工具/库 | 用途 |
|------|---------|------|
| 数据获取 | `tushare`, `akshare`, `requests` | 股票/新闻数据采集 |
| 文本处理 | `jieba` | 中文分词 |
| 情感分析 | `transformers`, `openai`, `httpx` | FinBERT + LLM API |
| 数据分析 | `pandas`, `numpy` | 数据处理 |
| 统计建模 | `statsmodels`, `scipy` | VAR、Granger检验、OLS回归 |
| 可视化 | `matplotlib`, `seaborn` | 图表绘制 |
| 配置管理 | `pyyaml`, `python-dotenv` | YAML配置 + 环境变量 |
| 报告 | Jupyter Notebook, Markdown | 可复现分析 |

---

## 快速开始

### 1. 环境准备

```bash
conda create -n quant_research python=3.11
conda activate quant_research
pip install -r requirements.txt
```

### 2. 配置

编辑 `.env` 文件：

```
TUSHARE_TOKEN=你的tushare_token
LLM_BASE_URL=https://your-api-endpoint/v1
LLM_API_KEY=your_api_key
LLM_MODEL_NAME=your_model_name
```

编辑 `config.yaml` 调整参数（标的、日期范围、情感方法等）。

### 3. 运行

**完整流水线：**

```bash
python main.py --method dictionary     # 词典法（最快，baseline）
python main.py --method finbert        # 词典法 + FinBERT
python main.py --method llm            # 词典法 + LLM API
python main.py --method both           # 全部方法
python main.py --method llm --force    # 强制重跑，忽略断点文件
python main.py --skip-news --method dictionary  # 跳过数据采集，用本地数据
```

**新闻过滤（独立运行）：**

```bash
python scripts/filter_news.py          # 过滤新闻，生成 filtered_news.csv
python scripts/filter_news.py --force  # 强制重新过滤
```

### 4. 查看结果

- 运行输出：`outputs/runs/run00XX_YYYYMMDD_HHMMSS/`
- 图表：`outputs/runs/run00XX_.../figures/`
- 合并数据：`outputs/runs/run00XX_.../processed/merged_dataset.csv`
- 分析报告：`outputs/runs/run00XX_.../analysis_report.md`

---

## 预期假设

基于 Tetlock (2007) 和 Kirtac (2024) 的研究，我们预期：

1. **H1**：前一日负面新闻情绪显著预测当日A股负收益（短期价格压力）
2. **H2**：该效应在 3-5 个交易日内反转（支持噪声交易者假说）
3. **H3**：极端情绪日（无论正负）预测更高交易量
4. **H4**：LLM 情感分析的预测能力优于词典法
5. **H5**：政策类新闻的情感冲击大于公司层面新闻（A股特色）

---

## 参考资料

### 核心论文
- Tetlock, P.C. (2007). "Giving Content to Investor Sentiment: The Role of Media in the Stock Market." *Journal of Finance*, 62(3), 1139-1168.
- Kirtac, K. & Germano, G. (2024). "Sentiment trading with large language models." *Finance Research Letters*, 62, 105227.

### 推荐阅读
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Hilpisch, Y. (2018). *Python for Finance*. O'Reilly.
- Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35-65.

### 工具与数据
- [Tushare Pro](https://tushare.pro/) — A股数据接口
- [AkShare](https://akshare.akfamily.xyz/) — 开源财经数据接口
- [GDELT Project](https://www.gdeltproject.org/) — 全球新闻事件数据库
- [statsmodels](https://www.statsmodels.org/) — 统计模型与因果检验
- [HuggingFace Transformers](https://huggingface.co/transformers/) — NLP 模型库
- [FNSPID Dataset](https://arxiv.org/abs/2402.06698) — 金融新闻-股价时间序列数据集
