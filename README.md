# 基于新闻文本的A股股价波动预测与因果关系初探

> 探索新闻文本与A股股票价格波动之间的关系，检验新闻情绪是否具有市场"预测能力"。

## 项目概述

本项目以 Tetlock (2007) 的经典研究"Giving Content to Investor Sentiment"为理论基石，结合 Kirtac & Germano (2024) 关于大语言模型在金融情感分析中的前沿成果，针对中国A股市场构建一套完整的"新闻情绪→股价预测"研究框架。

核心问题：**中文财经新闻的情绪指标能否预测A股个股或指数的未来收益与波动？**

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
├── README.md                  # 本文件
├── .env                       # 敏感配置（Tushare token、LLM API key）— 不提交 git
├── .gitignore
├── config.yaml                # 所有可调参数（标的、日期、模型参数、VAR配置等）
├── requirements.txt           # Python 依赖
├── main.py                    # 命令行入口（python main.py --method both）
├── papers/                    # 参考文献
│   ├── Tetlock-GivingContentInvestor-2007.pdf
│   └── 2412.19245v1.pdf
├── data/
│   ├── raw/                   # 原始数据（自动保存）
│   │   ├── stock_data.csv
│   │   └── news_data.csv
│   └── processed/             # 处理后数据
│       └── merged_dataset.csv
├── src/                       # 源代码
│   ├── __init__.py
│   ├── utils.py               # 配置加载、环境变量管理
│   ├── data_collector.py      # Tushare行情 + AkShare/GDELT新闻采集
│   ├── news_processor.py      # 中文分词(jieba)、文本清洗、GDELT解析
│   ├── sentiment_analyzer.py  # 词典法 + FinBERT + LLM API 三轨情感分析
│   ├── feature_engineer.py    # 日度情感指数、衰减加权、哑变量构造
│   ├── analysis.py            # VAR(5阶lag+星期/月份哑变量)、Granger检验、回归
│   └── visualization.py       # 行情图、情感图、IRF脉冲响应、仪表板
├── notebooks/
│   └── main_analysis.ipynb    # 端到端分析 Notebook
└── results/
    ├── figures/               # 自动生成的图表
    └── tables/                # 结果表格
```

### 配置文件说明

**`.env`**（敏感信息，不提交 git）：
```
TUSHARE_TOKEN=your_token_here
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=gpt-3.5-turbo
```

**`config.yaml`**（所有可调参数）：
- `data.index_code`: 目标指数代码（默认 `399300.SZ` 沪深300）
- `data.start_date / end_date`: 日期范围
- `sentiment.method`: 情感分析方法（`dictionary` / `finbert` / `llm` / `both`）
- `features.decay_lambda`: 情感衰减系数（默认 0.5）
- `var_model.lags`: VAR 滞后阶数（默认 5，与 Tetlock 一致）
- `var_model.weekday_dummies / month_dummies`: 是否加入哑变量

---

## 数据方案

### 股票数据

| 数据源 | 接口 | 内容 | 积分要求 |
|--------|------|------|---------|
| **Tushare** | `index_daily` | 沪深300日频行情（OHLCV） | 基础接口，2000积分可用 |
| **AkShare** | 备用 | 行情补充 | 免费 |

### 新闻数据

| 数据源 | 说明 | 优先级 |
|--------|------|--------|
| **AkShare** | 东方财富/新浪财经财经新闻 | 主力（免费、中文） |
| **GDELT** | 全球新闻事件数据库（BigQuery / 直接下载） | 补充（覆盖广） |

> **注意**：Tushare 新闻接口（`news`、`cctv_news`、`major_news`）需要 2000+ 以上积分，学生账号不可用。因此新闻数据优先使用 AkShare，不足时切换 GDELT。

### 数据时间范围

建议覆盖 **2018-2024年**（约6年），理由：
- 足够长的样本期以覆盖不同市场周期（牛市/熊市/震荡市）
- 与 LLM 论文的数据期（2010-2023）有重叠，便于方法对比
- 数据量适中，便于计算

---

## 方法论

### Phase 1: 数据采集与预处理

```
股票数据 → 日频 OHLCV + 收益率计算
新闻数据 → 文本清洗 → 去重 → 日期对齐
```

**关键步骤**：
1. 选取目标标的（建议：沪深300成分股 或 单只龙头股如贵州茅台）
2. 获取日频行情数据，计算日收益率 `return_t = (close_t - close_{t-1}) / close_{t-1}`
3. 采集对应时间段的新闻数据，按日期聚合
4. 新闻去重（基于文本相似度，参考 Kirtac 论文中 cosine similarity ≥ 0.8 的去重策略）

### Phase 2: 新闻文本处理与情感分析

#### Baseline: 词典法（Dictionary-based）

参照 Tetlock 的方法，构建/使用中文金融情感词典：

- **中文金融情感词典**：大连理工情感词典、台湾大学情感词典（NTUSD）、或基于 Loughran-McDonald 词典的中文翻译版
- **计算方式**：对每日所有新闻，统计正面词数 `pos_t` 和负面词数 `neg_t`
- **情感得分**：`sentiment_t = (pos_t - neg_t) / (pos_t + neg_t + 1)`

#### 进阶: LLM法

- **FinBERT**（中文微调版）：使用 HuggingFace 上的中文金融预训练模型
- **ChatGLM / Qwen**：国产大模型，对中文理解更好
- **计算方式**：对每篇新闻进行情感分类（正面/负面/中性），然后按日聚合

#### 情感衰减处理（鼓励项）

非交易日新闻的影响应随时间衰减：
```
effective_sentiment_t = Σ sentiment_{t-k} × λ^k    (λ ∈ (0,1), 如 λ=0.5)
```

### Phase 3: 特征工程

构造以下指标：

| 指标 | 定义 | 说明 |
|------|------|------|
| `sentiment_t` | 当日平均情感得分 | 核心变量 |
| `news_count_t` | 当日新闻条数 | 新闻强度/关注度 |
| `sentiment_std_t` | 当日情感标准差 | 意见分歧度 |
| `extreme_sentiment_t` | \|sentiment_t\| 是否超过阈值 | 极端情绪标记 |
| `policy_news_ratio_t` | 政策类新闻占比 | 区分新闻类型 |

### Phase 4: 统计分析

#### 4.1 描述性统计与相关性分析
- 情感得分与收益率的散点图、相关系数
- 不同情绪分组的收益分布对比

#### 4.2 回归分析（核心）

**基准回归**（对应 Tetlock 的 VAR 框架简化版）：

```
return_t = α + β₁ × sentiment_{t-1} + β₂ × return_{t-1} + β₃ × volume_{t-1} + ε_t
```

- 检验 `β₁` 是否显著为负（悲观情绪预测下跌）
- 加入滞后项 `sentiment_{t-2}` 到 `sentiment_{t-5}` 检验反转效应

**扩展回归**：
```
return_t = α + Σ β_k × sentiment_{t-k} + controls + ε_t    (k = 1,...,5)
```

#### 4.3 Granger 因果检验（鼓励项）

```python
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(data[['return', 'sentiment']], maxlag=5)
```

检验：新闻情绪是否 Granger-cause 股票收益？反之是否成立？

#### 4.4 交易量分析（对应 Tetlock 发现）

```
volume_t = α + γ × |sentiment_{t-1}| + controls + ε_t
```

检验极端情绪是否预测高交易量。

### Phase 5: 因果推断讨论（核心要求）

本项目重点不在于复杂模型，而在于对**相关性与因果性差异**的深入讨论：

1. **伪因果识别**：是否存在遗漏变量同时影响新闻情绪和股价？（如宏观经济冲击）
2. **反向因果**：股价下跌 → 媒体报道更悲观 → 看似"悲观预测下跌"实为"下跌导致悲观"
3. **信息延迟**：新闻发布时间 vs. 市场反应时间，是否存在信息提前泄露？
4. **市场有效性**：如果情绪能预测收益，这是否违背弱式有效市场假说？
5. **控制变量**：加入市场收益率、行业收益率、波动率等控制变量后，结论是否稳健？

---

## 技术栈

| 类别 | 工具/库 | 用途 |
|------|---------|------|
| 数据获取 | `tushare`, `akshare`, `requests` | 股票/新闻数据采集 |
| 文本处理 | `jieba`, `snownlp`, `transformers` | 中文分词、情感分析 |
| 数据分析 | `pandas`, `numpy` | 数据处理 |
| 统计建模 | `statsmodels`, `scipy` | 回归、Granger检验 |
| 可视化 | `matplotlib`, `seaborn` | 图表绘制 |
| 机器学习(可选) | `scikit-learn`, `torch` | LLM情感分析 |
| 报告 | Jupyter Notebook | 可复现分析 |

---

## 实施路线

### Week 1: 数据基础
- [ ] 注册 Tushare / 安装 AkShare，获取目标股票日频数据
- [ ] 搭建 GDELT 新闻数据获取流程（或东方财富爬虫）
- [ ] 数据清洗、日期对齐、初步探索性分析

### Week 2: 情感分析
- [ ] 实现词典法情感分析（baseline）
- [ ] 构造每日情感指数、新闻强度等特征
- [ ] 描述性统计与可视化

### Week 3: 统计建模
- [ ] 实现基准回归模型（return ~ lagged_sentiment）
- [ ] Granger 因果检验
- [ ] 交易量分析
- [ ] 稳健性检验（不同滞后阶数、不同标的）

### Week 4: 进阶与报告
- [ ] （可选）尝试 LLM 情感分析，对比词典法效果
- [ ] 情感衰减模型
- [ ] 因果性讨论与伪因果分析
- [ ] 撰写方法说明文档（≤4页）
- [ ] 整理 Notebook，上传 GitHub

---

## 预期结果与假设

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

---

## 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate quant_research

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `.env` 文件，填入你的 Tushare Token：
```
TUSHARE_TOKEN=你的tushare_token
```

如需使用 LLM 情感分析，额外填写：
```
LLM_BASE_URL=https://your-api-endpoint/v1
LLM_API_KEY=your_api_key
LLM_MODEL_NAME=your_model_name
```

### 3. 运行

**命令行方式**（完整流程）：
```bash
python main.py --method both        # 词典法 + FinBERT + LLM
python main.py --method dictionary  # 仅词典法（最快）
python main.py --method finbert     # 词典法 + FinBERT
```

**Notebook 方式**（交互式分析）：
打开 `notebooks/main_analysis.ipynb`，逐单元格运行。

### 4. 查看结果

- 图表：`results/figures/`
- 数据：`data/processed/merged_dataset.csv`
- Notebook 中包含完整的分析结果和结论
