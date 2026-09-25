# -*- coding: utf-8 -*-

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

SRC_DIR    = PROJECT_ROOT / "src"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR    = REPORT_DIR / "figures"
TABLE_DIR  = REPORT_DIR / "tables"
DOCS_DIR   = PROJECT_ROOT / "docs"

# 原始数据文件
RAW_CSV = RAW_DIR / "train_user.csv"

# 自动创建输出目录（幂等：重复执行不报错）
for _d in (INTERIM_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- DuckDB（分析引擎 / OLAP） ----------
# 本地分析库文件，不存在会自动创建
DUCKDB_PATH = DATA_DIR / "taobao.duckdb"
# 清洗后的列存文件（加速后续所有查询）
PARQUET_PATH = INTERIM_DIR / "user_behavior.parquet"

# ---------- MySQL 8.0.42（结果仓储 / OLTP） ----------
MYSQL_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "password",
    "database": "taobao_behavior",
    "charset":  "utf8mb4",
}

# ---------- 业务常量 ----------
DATA_START_DATE = "2014-11-18"     # 数据起始日
DATA_END_DATE   = "2014-12-18"     # 数据结束日

# 双十二大促窗口（P4 做双重差分 DID 用）
PROMO_START_DATE = "2014-12-10"
PROMO_END_DATE   = "2014-12-12"

# 行为类型字典：数据集里是数字编码，报告里要转成中文
BEHAVIOR_MAP = {
    1: "点击",
    2: "收藏",
    3: "加购",
    4: "支付",
}