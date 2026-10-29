"""
统计分析模块
- VAR 模型（5阶滞后 + 星期/月份哑变量，对齐 Tetlock 2007）
- Granger 因果检验
- 回归分析（情感→收益、收益→情感、极端情感→交易量）
- 稳健性检验
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


class StatisticalAnalyzer:
    """统计分析器：VAR、Granger 因果检验、回归分析"""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        var_config = config.get("var_model", {})
        self.lags = var_config.get("lags", 5)
        self.endogenous_vars = var_config.get("endogenous_vars", ["return", "sentiment", "volume"])
        self.weekday_dummies = var_config.get("weekday_dummies", True)
        self.month_dummies = var_config.get("month_dummies", True)
        self.significance_level = var_config.get("significance_level", 0.05)

    # ================================================================
    # VAR 模型
    # ================================================================
    def fit_var(self, df: pd.DataFrame) -> dict:
        """
        拟合 VAR 模型
        内生变量: return, sentiment, volume
        外生变量: 星期哑变量 + 月份哑变量
        滞后阶数: 5（与 Tetlock 一致）

        返回: {
            "model": VAR 模型对象,
            "results": 拟合结果,
            "summary": 摘要字符串,
            "coefficients": 系数表,
            "granger": Granger 因果检验结果,
        }
        """
        # 选取内生变量
        endog = df[self.endogenous_vars].copy()
        endog = endog.dropna()

        if len(endog) < self.lags * 3:
            raise ValueError(f"样本量不足 ({len(endog)} 行)，至少需要 {self.lags * 3} 行")

        # 构建外生变量（哑变量）
        exog = self._build_exog(df.loc[endog.index])

        logger.info(f"拟合 VAR({self.lags}) 模型，样本量: {len(endog)}")
        logger.info(f"内生变量: {self.endogenous_vars}")
        logger.info(f"外生变量: {list(exog.columns) if exog is not None else '无'}")

        # 拟合 VAR
        model = VAR(endog, exog=exog)

        # 滞后阶数选择（信息准则）
        lag_selection = model.select_order(maxlags=10)
        logger.info(f"滞后阶数选择: AIC={lag_selection.aic}, BIC={lag_selection.bic}")

        # 用指定滞后阶数拟合
        results = model.fit(maxlags=self.lags, ic=None)

        # Granger 因果检验
        granger_results = self._granger_causality(endog, exog)

        return {
            "model": model,
            "results": results,
            "summary": str(results.summary()),
            "coefficients": self._extract_coefficients(results),
            "granger": granger_results,
            "lag_selection": lag_selection,
        }

    def _build_exog(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """构建外生变量矩阵（哑变量）"""
        exog_cols = []

        if self.weekday_dummies:
            # 只用 4 个星期哑变量（避免完全共线性），去掉 weekday_4（星期五）
            weekday_cols = [c for c in df.columns if c.startswith("weekday_") and c != "weekday_4"]
            exog_cols.extend(weekday_cols)

        if self.month_dummies:
            # 只用 11 个月份哑变量（避免完全共线性），去掉 month_12
            month_cols = [c for c in df.columns if c.startswith("month_") and c != "month_12"]
            exog_cols.extend(month_cols)

        if not exog_cols:
            return None

        available = [c for c in exog_cols if c in df.columns]
        if not available:
            return None

        return df[available]

    def _extract_coefficients(self, results) -> pd.DataFrame:
        """提取 VAR 系数表"""
        coefs = results.params
        return coefs

    # ================================================================
    # Granger 因果检验
    # ================================================================
    def _granger_causality(
        self, endog: pd.DataFrame, exog: pd.DataFrame | None = None
    ) -> dict:
        """
        Granger 因果检验
        检验: sentiment Granger-causes return? return Granger-causes sentiment?
        """
        var_names = list(endog.columns)
        granger_results = {}

        for cause in var_names:
            for effect in var_names:
                if cause == effect:
                    continue
                try:
                    test_data = endog[[effect, cause]].dropna()
                    gc_result = grangercausalitytests(
                        test_data,
                        maxlag=self.lags,
                        verbose=False,
                    )
                    # 取最大滞后阶数的 F 检验 p 值
                    # 兼容不同 statsmodels 版本的返回格式
                    lag_result = gc_result[self.lags]
                    if isinstance(lag_result, dict):
                        f_pvalue = lag_result.get("ssr_ftest", (None, 1.0))[1]
                    elif isinstance(lag_result, (list, tuple)):
                        f_pvalue = lag_result[0]["ssr_ftest"][1]
                    else:
                        # 某些版本返回嵌套结构
                        f_pvalue = 1.0
                    granger_results[f"{cause}_→_{effect}"] = {
                        "f_pvalue": f_pvalue,
                        "significant": f_pvalue < self.significance_level,
                    }
                except Exception as e:
                    logger.debug(f"Granger 检验 {cause}→{effect} 失败: {e}")
                    granger_results[f"{cause}_→_{effect}"] = {
                        "f_pvalue": np.nan,
                        "significant": False,
                    }

        return granger_results

    # ================================================================
    # 回归分析
    # ================================================================
    def regression_sentiment_to_return(
        self, df: pd.DataFrame, sentiment_col: str = "avg_sentiment"
    ) -> dict:
        """
        回归: return_t = α + β₁ × sentiment_{t-1} + controls + ε_t
        对应 Tetlock 的核心回归
        """
        data = df.copy()

        # 构造滞后情感
        for lag in range(1, self.lags + 1):
            data[f"{sentiment_col}_lag{lag}"] = data[sentiment_col].shift(lag)

        # 控制变量
        data["return_lag1"] = data["return"].shift(1)
        data["volume_detrend_lag1"] = data["volume_detrend"].shift(1)

        # 构建回归式
        formula_cols = [c for c in [
            f"{sentiment_col}_lag1",
            f"{sentiment_col}_lag2",
            f"{sentiment_col}_lag3",
            f"{sentiment_col}_lag4",
            f"{sentiment_col}_lag5",
            "return_lag1",
            "volume_detrend_lag1",
        ] if c in data.columns]

        # 添加哑变量
        if self.weekday_dummies:
            formula_cols.extend([c for c in data.columns if c.startswith("weekday_") and c != "weekday_4"])
        if self.month_dummies:
            formula_cols.extend([c for c in data.columns if c.startswith("month_") and c != "month_12"])

        data_reg = data[["return"] + formula_cols].dropna()

        if len(data_reg) < 20:
            logger.warning(f"回归样本量不足: {len(data_reg)}")
            return {"error": "样本量不足"}

        X = sm.add_constant(data_reg[formula_cols])
        y = data_reg["return"]

        # 使用 Newey-West 稳健标准误（HAC）
        model = sm.OLS(y, X)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        return {
            "results": results,
            "summary": results.summary().as_text(),
            "coefficients": results.params,
            "pvalues": results.pvalues,
            "r_squared": results.rsquared,
            "adj_r_squared": results.rsquared_adj,
        }

    def regression_return_to_sentiment(
        self, df: pd.DataFrame, sentiment_col: str = "avg_sentiment"
    ) -> dict:
        """
        反向回归: sentiment_t = α + β × return_{t-1} + controls + ε_t
        检验反向因果（收益→情感），对应 Tetlock Table III
        """
        data = df.copy()

        # 滞后收益
        for lag in range(1, self.lags + 1):
            data[f"return_lag{lag}"] = data["return"].shift(lag)

        formula_cols = [c for c in [
            "return_lag1", "return_lag2", "return_lag3", "return_lag4", "return_lag5",
        ] if c in data.columns]

        if self.weekday_dummies:
            formula_cols.extend([c for c in data.columns if c.startswith("weekday_") and c != "weekday_4"])
        if self.month_dummies:
            formula_cols.extend([c for c in data.columns if c.startswith("month_") and c != "month_12"])

        data_reg = data[[sentiment_col] + formula_cols].dropna()

        if len(data_reg) < 20:
            return {"error": "样本量不足"}

        X = sm.add_constant(data_reg[formula_cols])
        y = data_reg[sentiment_col]

        model = sm.OLS(y, X)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        return {
            "results": results,
            "summary": results.summary().as_text(),
            "coefficients": results.params,
            "pvalues": results.pvalues,
            "r_squared": results.rsquared,
        }

    def regression_extreme_to_volume(
        self, df: pd.DataFrame
    ) -> dict:
        """
        回归: volume_t = α + γ × |sentiment_{t-1}| + controls + ε_t
        检验极端情绪是否预测高交易量（对应 Tetlock 发现）
        """
        data = df.copy()

        # 使用绝对情感值（极端程度）
        abs_sentiment_cols = []
        for col in ["avg_sentiment", "avg_sentiment_decay"]:
            if col in data.columns:
                abs_col = f"{col}_abs"
                data[abs_col] = data[col].abs()
                data[f"{abs_col}_lag1"] = data[abs_col].shift(1)
                abs_sentiment_cols.append(f"{abs_col}_lag1")

        if "return_lag1" not in data.columns:
            data["return_lag1"] = data["return"].shift(1)

        formula_cols = abs_sentiment_cols + ["return_lag1"]

        if self.weekday_dummies:
            formula_cols.extend([c for c in data.columns if c.startswith("weekday_") and c != "weekday_4"])
        if self.month_dummies:
            formula_cols.extend([c for c in data.columns if c.startswith("month_") and c != "month_12"])

        dep_var = "volume_detrend"
        data_reg = data[[dep_var] + formula_cols].dropna()

        if len(data_reg) < 20:
            return {"error": "样本量不足"}

        X = sm.add_constant(data_reg[formula_cols])
        y = data_reg[dep_var]

        model = sm.OLS(y, X)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        return {
            "results": results,
            "summary": results.summary().as_text(),
            "coefficients": results.params,
            "pvalues": results.pvalues,
            "r_squared": results.rsquared,
        }

    # ================================================================
    # 脉冲响应分析
    # ================================================================
    def impulse_response(
        self, var_result: dict, steps: int = 10
    ) -> dict:
        """
        计算脉冲响应函数（IRF）
        分析一个变量的冲击对其他变量的动态影响
        """
        results = var_result["results"]
        irf = results.irf(steps)

        # 提取 IRF 数据（irfs 是 3D numpy 数组: [steps, n_vars, n_vars]）
        var_list = list(results.model.endog_names)
        irf_data = {}
        for i, cause in enumerate(var_list):
            for j, effect in enumerate(var_list):
                irf_data[f"{cause}_→_{effect}"] = irf.irfs[:, j, i]

        # 正交化脉冲响应（使用 Cholesky 分解）
        P = np.linalg.cholesky(results.sigma_u)
        orth_irf = results.irf(steps, var_decomp=P)
        orth_irf_data = {}
        for i, cause in enumerate(var_list):
            for j, effect in enumerate(var_list):
                orth_irf_data[f"{cause}_→_{effect}"] = orth_irf.irfs[:, j, i]

        return {
            "irf": irf_data,
            "orthogonal_irf": orth_irf_data,
            "steps": steps,
        }

    # ================================================================
    # 方差分解
    # ================================================================
    def variance_decomposition(
        self, var_result: dict, steps: int = 10
    ) -> dict:
        """方差分解：每个变量的预测误差方差中有多少来自其他变量的冲击"""
        results = var_result["results"]
        fevd = results.fevd(steps)

        var_list = list(results.model.endog_names)
        fevd_data = {}
        for i, var in enumerate(var_list):
            fevd_data[var] = {}
            for j, cause in enumerate(var_list):
                fevd_data[var][cause] = fevd.decomp[i, :, j]

        return fevd_data


def run_full_analysis(df: pd.DataFrame, config: dict | None = None) -> dict:
    """
    运行完整分析流程
    返回所有分析结果的字典
    """
    analyzer = StatisticalAnalyzer(config)
    results = {}

    # 1. VAR 模型
    logger.info("=" * 60)
    logger.info("1. VAR 模型分析")
    logger.info("=" * 60)
    try:
        var_result = analyzer.fit_var(df)
        results["var"] = var_result
        logger.info("\nVAR 模型摘要:\n%s", var_result["summary"][:1000])
    except Exception as e:
        logger.error(f"VAR 模型拟合失败: {e}")
        results["var"] = {"error": str(e)}

    # 2. Granger 因果检验
    logger.info("\n" + "=" * 60)
    logger.info("2. Granger 因果检验")
    logger.info("=" * 60)
    if "var" in results and "granger" in results["var"]:
        for pair, gc in results["var"]["granger"].items():
            sig = "显著" if gc["significant"] else "不显著"
            logger.info(f"  {pair}: p={gc['f_pvalue']:.4f} ({sig})")

    # 3. 回归: 情感→收益
    logger.info("\n" + "=" * 60)
    logger.info("3. 回归: 情感→收益")
    logger.info("=" * 60)
    # 查找可用的情感列（优先 avg_sentiment，其次各方法独立列）
    sentiment_cols_to_test = []
    if "avg_sentiment" in df.columns:
        sentiment_cols_to_test.append("avg_sentiment")
    for score_col in ["dict_score", "finbert_score", "llm_score"]:
        if score_col in df.columns and score_col not in sentiment_cols_to_test:
            sentiment_cols_to_test.append(score_col)

    for score_col in sentiment_cols_to_test:
        reg_result = analyzer.regression_sentiment_to_return(df, score_col)
        results[f"regression_{score_col}"] = reg_result
        if "summary" in reg_result:
            logger.info(f"\n{score_col} 回归摘要:\n%s", reg_result["summary"][:800])
        elif "error" in reg_result:
            logger.warning(f"  {score_col}: {reg_result['error']}")

    # 4. 回归: 收益→情感（反向因果）
    logger.info("\n" + "=" * 60)
    logger.info("4. 回归: 收益→情感（反向因果检验）")
    logger.info("=" * 60)
    for score_col in sentiment_cols_to_test:
        reg_result = analyzer.regression_return_to_sentiment(df, score_col)
        results[f"reverse_regression_{score_col}"] = reg_result
        if "summary" in reg_result:
            logger.info(f"\n{score_col} 反向回归摘要:\n%s", reg_result["summary"][:800])
        elif "error" in reg_result:
            logger.warning(f"  {score_col}: {reg_result['error']}")

    # 5. 回归: 极端情感→交易量
    logger.info("\n" + "=" * 60)
    logger.info("5. 回归: 极端情感→交易量")
    logger.info("=" * 60)
    vol_result = analyzer.regression_extreme_to_volume(df)
    results["regression_volume"] = vol_result
    if "summary" in vol_result:
        logger.info("\n交易量回归摘要:\n%s", vol_result["summary"][:800])
    elif "error" in vol_result:
        logger.warning(f"  交易量回归: {vol_result['error']}")

    # 6. 脉冲响应
    if "var" in results and "error" not in results["var"]:
        logger.info("\n" + "=" * 60)
        logger.info("6. 脉冲响应分析")
        logger.info("=" * 60)
        irf_result = analyzer.impulse_response(results["var"])
        results["irf"] = irf_result

        # 方差分解
        fevd_result = analyzer.variance_decomposition(results["var"])
        results["fevd"] = fevd_result

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 模拟数据测试
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)

    df = pd.DataFrame({
        "trade_date": dates,
        "return": np.random.randn(n) * 0.01,
        "avg_sentiment": np.random.randn(n) * 0.3,
        "volume_detrend": np.random.randn(n),
    })

    # 添加哑变量
    weekday = df["trade_date"].dt.weekday
    for i in range(5):
        df[f"weekday_{i}"] = (weekday == i).astype(int)
    month = df["trade_date"].dt.month
    for m in range(1, 13):
        df[f"month_{m}"] = (month == m).astype(int)

    results = run_full_analysis(df)
    print("\n分析完成！")
