"""
工具函数：配置加载、环境变量管理、日志、运行管理
"""
from __future__ import annotations

import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env", override=True)


# ============================================================
# 配置加载
# ============================================================
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


# ============================================================
# JSON 序列化辅助
# ============================================================
def convert_to_json_serializable(obj: Any) -> Any:
    """将对象转换为 JSON 可序列化的格式"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return {"__type__": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
    elif isinstance(obj, pd.Series):
        return {"__type__": "Series", "length": len(obj), "name": obj.name}
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)
    else:
        try:
            if hasattr(obj, 'item'):
                return obj.item()
        except Exception:
            pass
        return str(obj)


# ============================================================
# 运行管理：递增序号 + 时间戳，每次运行独立输出目录
# ============================================================
_current_run_id: Optional[str] = None
_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
_COUNTER_FILE = _RUNS_DIR / "run_counter.json"


def _load_counter() -> tuple[int, list]:
    """加载运行计数器，返回 (next_id, records)"""
    if _COUNTER_FILE.exists():
        try:
            with open(_COUNTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("next_id", 1), data.get("records", [])
        except Exception:
            pass
    return 1, []


def _save_counter(next_id: int, records: list):
    """保存运行计数器"""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"next_id": next_id, "records": records}
    tmp = _COUNTER_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.rename(_COUNTER_FILE)
    except Exception:
        with open(_COUNTER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def init_run(config: dict, run_id: int | None = None) -> str:
    """
    初始化一次运行。

    分配递增序号，生成 run<NNNN>_<YYYYMMDD_HHMMSS> 格式的目录名，
    保存 config.json（含完整参数、时间戳、git commit）。

    参数:
        config: 配置字典
        run_id: 可选的手动指定序号。不传则自动递增。

    返回:
        当前 run_id 字符串（如 "run0001_20261018_160000"）
    """
    global _current_run_id

    next_id, records = _load_counter()

    if run_id is not None:
        num = run_id
    else:
        num = next_id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run{num:04d}_{timestamp}"

    _current_run_id = run_dir_name
    run_dir = get_run_dir()
    ensure_dir(run_dir)

    # 保存运行参数
    run_meta = {
        "run_id": num,
        "run_dir": run_dir_name,
        "timestamp": datetime.now().isoformat(),
        "config": convert_to_json_serializable(config),
    }

    # 尝试记录 git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=5
        )
        if result.returncode == 0:
            run_meta["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 更新计数器
    if run_id is None:
        next_id = num + 1
    else:
        next_id = max(next_id, num + 1)
    records.append({"run_id": num, "run_dir": run_dir_name, "timestamp": timestamp})
    _save_counter(next_id, records)

    return run_dir_name


def get_run_dir() -> Path | None:
    """获取当前运行的输出目录；未初始化 run 则返回 None"""
    if _current_run_id is None:
        return None
    return PROJECT_ROOT / "outputs" / "runs" / _current_run_id


# ============================================================
# 日志管理
# ============================================================
def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    创建并配置日志记录器。

    日志同时输出到控制台和 run.log 文件（如果 run 目录已初始化）。
    如果 run 目录在 logger 创建之后才初始化，再次调用本函数会自动追加文件 handler。

    参数:
        name: 日志记录器名称（通常传模块名）
        log_level: 日志级别，如 "DEBUG", "INFO", "WARNING"

    返回:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 handler（只添加一次）
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                      for h in logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件 handler（如果 run 目录已初始化，且尚未添加文件 handler）
    run_dir = get_run_dir()
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    if run_dir is not None and not has_file:
        log_file = run_dir / "run.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# 路径工具
# ============================================================
def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在则自动创建"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(subfolder: str = "raw") -> Path:
    """获取数据目录路径"""
    return PROJECT_ROOT / "data" / subfolder


def get_output_path(subfolder: str = "results") -> Path:
    """
    获取输出目录路径。
    如果已初始化 run，路径包含 run_id；否则回退到项目根目录下的 outputs/。
    """
    run_dir = get_run_dir()
    if run_dir is not None:
        return run_dir / subfolder
    return PROJECT_ROOT / "outputs" / subfolder
