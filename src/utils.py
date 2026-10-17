"""
工具函数：配置加载、环境变量管理
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_path: str | Path | None = None) -> dict:
    """加载 config.yaml 配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tushare_token() -> str:
    """获取 Tushare API Token"""
    token = os.getenv("TUSHARE_TOKEN")
    if not token or token == "your_tushare_token_here":
        raise ValueError(
            "请在 .env 文件中设置 TUSHARE_TOKEN。\n"
            f".env 文件路径: {PROJECT_ROOT / '.env'}"
        )
    return token


def get_llm_config() -> dict:
    """获取 LLM API 配置"""
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")
    model_name = os.getenv("LLM_MODEL_NAME", "")

    if base_url == "https://api.example.com/v1":
        base_url = ""

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }
