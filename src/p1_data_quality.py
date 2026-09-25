# -*- coding: utf-8 -*-
"""
P1 data quality diagnostics and warehouse modeling.

Loads the raw CSV with DuckDB, runs a set of quality checks, writes a
cleaned Parquet copy, builds the DWD/DWS layers, and exports the daily
metrics to MySQL.

Deduplication policy
--------------------
No row-level deduplication is applied. Investigation (p1_check_dup.py,
p1_dup_test.py) showed the repeated records are genuine user behavior,
not logging duplication:

  - repeat count distribution decays smoothly, max = 22
    (a logging bug would produce a long tail of extreme values)
  - 99.69% of users have repeats
  - hourly repeat rate is flat (47%-50%) with no high-traffic peaks
    (a logging bug would concentrate during peak hours)

The source is already an hour-granular event stream, so repeated
(user, item, behavior, hour) rows are the dataset's given representation
of user activity. Only structurally invalid rows are removed.

Usage:
    python src/p1_data_quality.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68

NULL_COLS = [
    ("user_id缺失", "user_id"),
    ("item_id缺失", "item_id"),
    ("behavior_type缺失", "behavior_type"),
    ("geohash缺失", "user_geohash"),
    ("category缺失", "item_category"),
    ("time缺失", "time"),
]


class Tee:
    """Mirror stdout to a log file."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def show(df, max_rows=40):
    if df is None or len(df) == 0:
        print("    (empty)")
        return
    import pandas as pd
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))


def main():
    t0 = time.time()
    log_path = config.TABLE_DIR / "P1_运行日志.txt"
    sys.stdout = Tee(log_path)

    print(DIVIDER)
    print("  P1 数据质量诊断与数仓建模")
    print(DIVIDER)
    print(f"  数据文件  : {config.RAW_CSV}")
    print(f"  项目根目录: {config.PROJECT_ROOT}")
    print(f"  DuckDB    : {duckdb.__version__}")

    if not config.RAW_CSV.exists():
        print(f"\n[ERROR] 数据文件不存在: {config.RAW_CSV}")
        return

    size_mb = config.RAW_CSV.stat().st_size / 1024 / 1024
    print(f"  文件大小  : {size_mb:.1f} MB")

    con = duckdb.connect(str(config.DUCKDB_PATH))
    print(f"  DuckDB 库 : {config.DUCKDB_PATH}")

    csv_path = config.RAW_CSV.as_posix()
    raw = f"read_csv_auto('{csv_path}', header=true)"

    # ---- 0. schema ----
    section("【第 0 步】表结构探查")
    schema = con.execute(f"DESCRIBE SELECT * FROM {raw}").fetchdf()
    show(schema)

    # ---- 1. scale ----
    section("【第 1 步】数据规模诊断")
    scale = con.execute(f"""
        SELECT COUNT(*)                      AS 总记录数,
               COUNT(DISTINCT user_id)       AS 用户数,
               COUNT(DISTINCT item_id)       AS 商品数,
               COUNT(DISTINCT item_category) AS 类目数
        FROM {raw}
    """).fetchdf()
    show(scale)

    total = int(scale.loc[0, "总记录数"])
    real_users = int(scale.loc[0, "用户数"])
    print(f"\n  人均行为数: {total / real_users:.1f} 条/人")

    # ---- 2. time ----
    section("【第 2 步】时间字段诊断")
    show(con.execute(f"""
        SELECT MIN(time) AS 最早时间, MAX(time) AS 最晚时间,
               COUNT(DISTINCT time) AS 不同时间取值数
        FROM {raw}
    """).fetchdf())

    print("\n  按天统计记录数：")
    show(con.execute(f"""
        SELECT LEFT(time, 10) AS 日期, COUNT(*) AS 记录数,
               CASE WHEN LEFT(time, 10) BETWEEN '{config.PROMO_START_DATE}'
                    AND '{config.PROMO_END_DATE}' THEN '*' ELSE '' END AS 大促
        FROM {raw} GROUP BY 1, 3 ORDER BY 1
    """).fetchdf(), max_rows=40)

    # ---- 3. behavior ----
    section("【第 3 步】行为类型分布")
    show(con.execute(f"""
        SELECT behavior_type AS 行为编码,
               CASE behavior_type WHEN 1 THEN '点击' WHEN 2 THEN '收藏'
                    WHEN 3 THEN '加购' WHEN 4 THEN '支付'
                    ELSE '异常值' END AS 行为名称,
               COUNT(*) AS 事件数,
               COUNT(DISTINCT user_id) AS 涉及用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) AS 事件占比_百分比
        FROM {raw} GROUP BY 1, 2 ORDER BY 1
    """).fetchdf())

    # ---- 4. nulls ----
    section("【第 4 步】缺失值诊断")
    nulls = con.execute(f"""
        SELECT COUNT(*) AS 总行数,
               SUM(CASE WHEN user_id       IS NULL THEN 1 ELSE 0 END) AS "user_id缺失",
               SUM(CASE WHEN item_id       IS NULL THEN 1 ELSE 0 END) AS "item_id缺失",
               SUM(CASE WHEN behavior_type IS NULL THEN 1 ELSE 0 END) AS "behavior_type缺失",
               SUM(CASE WHEN user_geohash  IS NULL THEN 1 ELSE 0 END) AS "geohash缺失",
               SUM(CASE WHEN item_category IS NULL THEN 1 ELSE 0 END) AS "category缺失",
               SUM(CASE WHEN time          IS NULL THEN 1 ELSE 0 END) AS "time缺失"
        FROM {raw}
    """).fetchdf()
    show(nulls)

    print("\n  缺失率明细：")
    for alias, label in NULL_COLS:
        if alias not in nulls.columns:
            continue
        n = nulls.loc[0, alias]
        n = 0 if n is None else int(n)
        print(f"    {label:<16} {n:>12,} 行   {n * 100.0 / total:>7.3f}%")

    # ---- 5. repeat records ----
    section("【第 5 步】重复记录诊断")
    print("  说明：本数据集为小时粒度事件流，重复记录经检验为真实用户行为")
    print("        （依据见 p1_dup_test.py），因此不做行级去重。")
    print("        此处仅统计重复规模，供报告披露。\n")

    dup = con.execute(f"""
        WITH grp AS (
            SELECT user_id, item_id, behavior_type, time, COUNT(*) AS c
            FROM {raw} GROUP BY 1, 2, 3, 4
        )
        SELECT (SELECT COUNT(*) FROM grp) AS 唯一组合数,
               (SELECT SUM(c)   FROM grp) AS 总记录数,
               (SELECT COALESCE(SUM(c - 1), 0) FROM grp WHERE c > 1) AS 重复行数,
               (SELECT COUNT(*) FROM grp WHERE c > 1) AS 重复组合数,
               (SELECT MAX(c)   FROM grp) AS 最大重复次数
        FROM (SELECT 1)
    """).fetchdf()
    show(dup)

    dup_rows = int(dup.loc[0, "重复行数"])
    unique_comb = int(dup.loc[0, "唯一组合数"])
    share_dup = dup_rows * 100.0 / total
    print(f"\n  (user,item,behavior,hour) 唯一组合数 : {unique_comb:,}")
    print(f"  同组合多次出现的行数占比            : {share_dup:.4f}%")
    print(f"  判定：真实行为，保留全部记录")

    # ---- 6. value validity ----
    section("【第 6 步】值合理性诊断")
    show(con.execute(f"""
        SELECT (SELECT COUNT(*) FROM {raw} WHERE behavior_type NOT IN (1,2,3,4)) AS 行为编码越界,
               (SELECT COUNT(*) FROM {raw} WHERE user_id <= 0)       AS 用户ID非正,
               (SELECT COUNT(*) FROM {raw} WHERE item_id <= 0)       AS 商品ID非正,
               (SELECT COUNT(*) FROM {raw} WHERE item_category <= 0) AS 类目ID非正,
               (SELECT COUNT(*) FROM {raw}
                WHERE time < '{config.DATA_START_DATE}'
                   OR time > '{config.DATA_END_DATE} 23')             AS 时间越界
    """).fetchdf())

    # ---- 7. geohash usability ----
    section("【第 7 步】user_geohash 可用性评估")
    geo = con.execute(f"""
        SELECT COUNT(*) AS 总行数,
               SUM(CASE WHEN user_geohash IS NOT NULL THEN 1 ELSE 0 END) AS 有值行数,
               COUNT(DISTINCT CASE WHEN user_geohash IS NOT NULL
                                   THEN user_geohash END) AS 不同geohash数
        FROM {raw}
    """).fetchdf()
    show(geo)

    geo_rows = int(geo.loc[0, "有值行数"])
    geo_pct = geo_rows * 100.0 / total
    print(f"\n  geohash 行覆盖率 : {geo_pct:.3f}%")
    print(f"  不同 geohash 数量: {int(geo.loc[0, '不同geohash数']):,}")

    user_geo = con.execute(f"""
        WITH per_user AS (
            SELECT user_id,
                   MAX(CASE WHEN user_geohash IS NOT NULL THEN 1 ELSE 0 END) AS has_geo
            FROM {raw} GROUP BY 1
        )
        SELECT COUNT(*) AS 用户总数,
               SUM(has_geo) AS 可定位用户数,
               ROUND(SUM(has_geo) * 100.0 / COUNT(*), 2) AS 用户覆盖率_百分比
        FROM per_user
    """).fetchdf()
    print("\n  用户级覆盖率：")
    show(user_geo)

    user_cov = float(user_geo.loc[0, "用户覆盖率_百分比"])
    print(f"\n  行级覆盖率 {geo_pct:.2f}% vs 用户级覆盖率 {user_cov:.2f}%")

    # geohash stability within one hour
    geo_stable = con.execute(f"""
        WITH grp AS (
            SELECT user_id, item_id, behavior_type, time,
                   COUNT(*) AS n_row,
                   COUNT(DISTINCT user_geohash) AS n_geo
            FROM {raw} GROUP BY 1, 2, 3, 4
        )
        SELECT SUM(CASE WHEN n_row > 1 AND n_geo > 1 THEN 1 ELSE 0 END) AS 同小时内多地理组合数,
               SUM(CASE WHEN n_row > 1 THEN 1 ELSE 0 END)              AS 多行组合数
        FROM grp
    """).fetchdf()
    print("\n  geohash 稳定性检查（同一小时内的多个不同地理编码）：")
    show(geo_stable)

    # ---- 8. behavior depth ----
    section("【第 8 步】用户行为深度分布")
    show(con.execute(f"""
        WITH per_user AS (SELECT user_id, COUNT(*) AS cnt FROM {raw} GROUP BY 1)
        SELECT COUNT(*) AS 用户总数, ROUND(AVG(cnt), 2) AS 人均行为数,
               MEDIAN(cnt) AS 中位数, MIN(cnt) AS 最小值, MAX(cnt) AS 最大值,
               SUM(CASE WHEN cnt = 1 THEN 1 ELSE 0 END) AS 仅1条行为的用户数
        FROM per_user
    """).fetchdf())

    print("\n  行为数分位数：")
    show(con.execute(f"""
        WITH per_user AS (SELECT user_id, COUNT(*) AS cnt FROM {raw} GROUP BY 1)
        SELECT ROUND(QUANTILE_CONT(cnt, 0.25), 0) AS p25,
               ROUND(QUANTILE_CONT(cnt, 0.50), 0) AS p50中位数,
               ROUND(QUANTILE_CONT(cnt, 0.75), 0) AS p75,
               ROUND(QUANTILE_CONT(cnt, 0.90), 0) AS p90,
               ROUND(QUANTILE_CONT(cnt, 0.99), 0) AS p99
        FROM per_user
    """).fetchdf())

    # ---- 9. quality report ----
    section("【第 9 步】导出数据质量报告")
    import pandas as pd
    report_rows = []

    def add(item, value, note=""):
        report_rows.append({"检查项": item, "结果": value, "说明": note})

    add("总记录数", f"{total:,}", "CSV 原始行数")
    add("用户数(UV)", f"{real_users:,}", "COUNT(DISTINCT user_id)")
    add("商品数", f"{int(scale.loc[0, '商品数']):,}", "")
    add("类目数", f"{int(scale.loc[0, '类目数']):,}", "")
    add("文件大小", f"{size_mb:.1f} MB", "")
    add("时间范围", f"{config.DATA_START_DATE} ~ {config.DATA_END_DATE}", "31 天")
    add("时间精度", "小时", "无法做 30 分钟会话切分")
    add("重复记录占比", f"{share_dup:.2f}%", "经检验为真实行为，保留")
    add("geohash 行覆盖率", f"{geo_pct:.3f}%", "")
    add("geohash 用户覆盖率", f"{user_cov:.2f}%", "")
    add("去重策略", "不去重", "小时粒度事件流")

    report_df = pd.DataFrame(report_rows)
    out_path = config.TABLE_DIR / "P1_数据质量报告.csv"
    report_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已导出: {out_path}")

    # ---- 10. clean -> parquet ----
    section("【第 10 步】清洗并生成 Parquet 列存")
    print("  规则: 行为编码合法 / user_id>0 / item_id>0 / 时间在范围内")
    print("  不去重（见第 5 步说明）")

    con.execute(f"""
        COPY (
            SELECT
                CAST(user_id       AS BIGINT)        AS user_id,
                CAST(item_id       AS BIGINT)        AS item_id,
                CAST(behavior_type AS TINYINT)       AS behavior_type,
                NULLIF(TRIM(user_geohash), '')       AS user_geohash,
                CAST(item_category AS INTEGER)       AS item_category,
                CAST(time AS VARCHAR)                AS time_raw,
                CAST(LEFT(time, 10) AS DATE)         AS dt,
                CAST(SUBSTR(time, 12, 2) AS TINYINT) AS hr,
                (SUBSTR(time, 12, 2) || ':00:00')::TIME AS ts,
                CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END AS is_buy
            FROM {raw}
            WHERE behavior_type IN (1, 2, 3, 4)
              AND user_id > 0 AND item_id > 0
              AND time >= '{config.DATA_START_DATE}'
              AND time <= '{config.DATA_END_DATE} 23'
        ) TO '{config.PARQUET_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    parquet_mb = config.PARQUET_PATH.stat().st_size / 1024 / 1024
    parquet = f"read_parquet('{config.PARQUET_PATH.as_posix()}')"
    clean_cnt = con.execute(f"SELECT COUNT(*) FROM {parquet}").fetchone()[0]
    print(f"  Parquet : {config.PARQUET_PATH}")
    print(f"  行数    : {clean_cnt:,}")
    print(f"  大小    : {parquet_mb:.1f} MB（原始 {size_mb:.1f} MB，压缩率 "
          f"{parquet_mb * 100.0 / size_mb:.1f}%）")

    # ---- 11. DWD ----
    section("【第 11 步】DWD 层：用户行为事实表")
    con.execute("DROP TABLE IF EXISTS dwd_user_behavior")
    con.execute(f"""
        CREATE TABLE dwd_user_behavior AS
        SELECT user_id, item_id, item_category, behavior_type,
               user_geohash, dt, hr, ts, time_raw, is_buy
        FROM {parquet}
    """)
    print(f"  dwd_user_behavior: "
          f"{con.execute('SELECT COUNT(*) FROM dwd_user_behavior').fetchone()[0]:,} 行")

    # ---- 12. DWS ----
    section("【第 12 步】DWS 层：每日核心指标表")
    con.execute("DROP TABLE IF EXISTS dws_daily_metrics")
    con.execute("""
        CREATE TABLE dws_daily_metrics AS
        WITH base AS (
            SELECT dt, COUNT(*) AS pv, COUNT(DISTINCT user_id) AS uv,
                   COUNT(DISTINCT item_id) AS item_cnt,
                   COUNT(DISTINCT item_category) AS category_cnt
            FROM dwd_user_behavior GROUP BY dt
        ),
        beh AS (
            SELECT dt,
                   COUNT(DISTINCT CASE WHEN behavior_type = 1 THEN user_id END) AS click_uv,
                   COUNT(DISTINCT CASE WHEN behavior_type = 2 THEN user_id END) AS fav_uv,
                   COUNT(DISTINCT CASE WHEN behavior_type = 3 THEN user_id END) AS cart_uv,
                   COUNT(DISTINCT CASE WHEN behavior_type = 4 THEN user_id END) AS buy_uv,
                   COUNT(CASE WHEN behavior_type = 2 THEN 1 END) AS fav_pv,
                   COUNT(CASE WHEN behavior_type = 3 THEN 1 END) AS cart_pv,
                   COUNT(CASE WHEN behavior_type = 4 THEN 1 END) AS buy_pv
            FROM dwd_user_behavior GROUP BY dt
        )
        SELECT b.dt, b.pv, b.uv, b.item_cnt, b.category_cnt,
               ROUND(b.pv * 1.0 / NULLIF(b.uv, 0), 3) AS pv_per_uv,
               e.click_uv, e.fav_uv, e.cart_uv, e.buy_uv,
               e.fav_pv, e.cart_pv, e.buy_pv,
               ROUND(e.buy_uv * 100.0 / NULLIF(b.uv, 0), 4) AS pay_rate_dau,
               ROUND(e.buy_uv * 100.0 / NULLIF(e.click_uv, 0), 4) AS pay_rate_click
        FROM base b LEFT JOIN beh e ON b.dt = e.dt
        ORDER BY b.dt
    """)
    daily = con.execute("SELECT * FROM dws_daily_metrics ORDER BY dt").fetchdf()
    show(daily, max_rows=40)

    # ---- 13. dim_date ----
    section("【第 13 步】日期维度表 dim_date")
    con.execute("DROP TABLE IF EXISTS dim_date")
    con.execute(f"""
        CREATE TABLE dim_date AS
        SELECT d::DATE AS dt,
               EXTRACT(YEAR  FROM d)::INTEGER AS year,
               EXTRACT(MONTH FROM d)::INTEGER AS month,
               EXTRACT(DAY   FROM d)::INTEGER AS day,
               EXTRACT(DOW   FROM d)::INTEGER AS dow,
               CASE EXTRACT(DOW FROM d) WHEN 0 THEN '周日' WHEN 1 THEN '周一'
                    WHEN 2 THEN '周二' WHEN 3 THEN '周三' WHEN 4 THEN '周四'
                    WHEN 5 THEN '周五' ELSE '周六' END AS weekday_cn,
               CASE WHEN EXTRACT(DOW FROM d) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
               CASE WHEN d::DATE BETWEEN DATE '{config.PROMO_START_DATE}'
                                      AND DATE '{config.PROMO_END_DATE}'
                    THEN 1 ELSE 0 END AS is_promo
        FROM generate_series(DATE '{config.DATA_START_DATE}',
                             DATE '{config.DATA_END_DATE}',
                             INTERVAL 1 DAY) t(d)
        ORDER BY dt
    """)
    print(f"  dim_date: "
          f"{con.execute('SELECT COUNT(*) FROM dim_date').fetchone()[0]} 行")

    # ---- 14. MySQL ----
    section("【第 14 步】MySQL 建库建表并写入")
    xlsx_path = config.TABLE_DIR / "P1_检查结果.xlsx"
    try:
        from sqlalchemy import create_engine, text
        cfg = config.MYSQL_CONFIG

        engine_server = create_engine(
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/?charset=utf8mb4")
        with engine_server.connect() as conn:
            print(f"  MySQL 版本: {conn.execute(text('SELECT VERSION()')).scalar()}")
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"))
            conn.commit()
            print(f"  数据库 `{cfg['database']}` 已就绪")

        engine = create_engine(
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?charset=utf8mb4")

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dws_daily_metrics (
                    dt              DATE          NOT NULL COMMENT '统计日期',
                    pv              BIGINT        NULL     COMMENT '页面浏览事件数',
                    uv              BIGINT        NULL     COMMENT '独立访客数',
                    item_cnt        BIGINT        NULL     COMMENT '涉及商品数',
                    category_cnt    BIGINT        NULL     COMMENT '涉及类目数',
                    pv_per_uv       DECIMAL(18,3) NULL     COMMENT '人均浏览深度',
                    click_uv        BIGINT        NULL     COMMENT '点击用户数',
                    fav_uv          BIGINT        NULL     COMMENT '收藏用户数',
                    cart_uv         BIGINT        NULL     COMMENT '加购用户数',
                    buy_uv          BIGINT        NULL     COMMENT '支付用户数',
                    fav_pv          BIGINT        NULL     COMMENT '收藏事件数',
                    cart_pv         BIGINT        NULL     COMMENT '加购事件数',
                    buy_pv          BIGINT        NULL     COMMENT '支付事件数',
                    pay_rate_dau    DECIMAL(10,4) NULL     COMMENT '付费率(支付UV/日活UV)',
                    pay_rate_click  DECIMAL(10,4) NULL     COMMENT '付费率(支付UV/点击UV)',
                    PRIMARY KEY (dt)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                  COLLATE=utf8mb4_general_ci COMMENT='每日核心指标表'
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dim_date (
                    dt          DATE       NOT NULL COMMENT '日期',
                    year        SMALLINT   NULL,
                    month       TINYINT    NULL,
                    day         TINYINT    NULL,
                    dow         TINYINT    NULL COMMENT '星期(0=周日)',
                    weekday_cn  VARCHAR(8) NULL,
                    is_weekend  TINYINT    NULL,
                    is_promo    TINYINT    NULL COMMENT '是否双十二窗口',
                    PRIMARY KEY (dt)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                  COLLATE=utf8mb4_general_ci COMMENT='日期维度表'
            """))
        print("  表结构已创建")

        daily_mysql = daily.copy()
        daily_mysql["dt"] = pd.to_datetime(daily_mysql["dt"]).dt.date
        daily_mysql.to_sql("dws_daily_metrics", engine, if_exists="replace",
                           index=False, method="multi", chunksize=500)
        print(f"  dws_daily_metrics 写入 {len(daily_mysql)} 行")

        dim_df = con.execute("SELECT * FROM dim_date ORDER BY dt").fetchdf()
        dim_df["dt"] = pd.to_datetime(dim_df["dt"]).dt.date
        dim_df.to_sql("dim_date", engine, if_exists="replace",
                      index=False, method="multi", chunksize=500)
        print(f"  dim_date 写入 {len(dim_df)} 行")

        with engine.connect() as conn:
            chk = conn.execute(text(
                "SELECT COUNT(*), MIN(dt), MAX(dt) FROM dws_daily_metrics")).fetchone()
            print(f"  回读验证: {chk[0]} 行, {chk[1]} ~ {chk[2]}")

    except Exception as e:
        print(f"  [ERROR] MySQL: {type(e).__name__}: {str(e)[:400]}")

    # ---- 15. excel export ----
    section("【第 15 步】导出结果到 Excel")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            schema.to_excel(writer, sheet_name="表结构", index=False)
            pd.DataFrame(report_rows).to_excel(writer, sheet_name="质量报告", index=False)
            con.execute("SELECT * FROM dim_date ORDER BY dt").fetchdf().to_excel(
                writer, sheet_name="日期维度", index=False)
            daily.to_excel(writer, sheet_name="每日指标", index=False)
            con.execute(f"""
                SELECT LEFT(time,10) AS 日期, COUNT(*) AS 记录数 FROM {raw}
                GROUP BY 1 ORDER BY 1
            """).fetchdf().to_excel(writer, sheet_name="每日记录数", index=False)
        print(f"  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal

    section("P1 执行完成")
    print(f"  总耗时  : {time.time() - t0:.1f} 秒")
    print(f"  运行日志: {log_path}")
    print(f"  Excel   : {xlsx_path}")
    print(f"  Parquet : {config.PARQUET_PATH}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
