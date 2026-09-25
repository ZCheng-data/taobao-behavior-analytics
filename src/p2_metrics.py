# -*- coding: utf-8 -*-
"""
P2 metrics layer: user daily metrics, user profiles, funnel, retention, RFM.

Builds the DWS/ADS tables on top of dwd_user_behavior produced by P1.

Prerequisites:
    - P1 has run: data/interim/user_behavior.parquet exists

Usage:
    python src/p2_metrics.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68


class Tee:
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
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))


def main():
    t0 = time.time()
    log_path = config.TABLE_DIR / "P2_运行日志.txt"
    sys.stdout = Tee(log_path)

    print(DIVIDER)
    print("  P2 指标体系建设")
    print(DIVIDER)

    if not config.PARQUET_PATH.exists():
        print(f"[ERROR] 未找到 {config.PARQUET_PATH}，请先运行 p1_data_quality.py")
        return

    con = duckdb.connect(str(config.DUCKDB_PATH))
    print(f"  DuckDB : {con.execute('SELECT version()').fetchone()[0]}")
    n_src = con.execute("SELECT COUNT(*) FROM dwd_user_behavior").fetchone()[0]
    print(f"  明细行 : {n_src:,}")

    # ==========================================================
    # 【第 1 步】用户日粒度指标表
    # ==========================================================
    section("【第 1 步】dws_user_daily —— 用户日粒度指标表")
    print("  粒度：一个用户 × 一天 一行")
    print("  用途：漏斗、留存、RFM 都从这张表出发\n")

    con.execute("DROP TABLE IF EXISTS dws_user_daily")
    con.execute("""
        CREATE TABLE dws_user_daily AS
        SELECT
            user_id,
            dt,
            COUNT(*)                                            AS pv,
            COUNT(DISTINCT item_id)                             AS item_cnt,
            COUNT(CASE WHEN behavior_type = 1 THEN 1 END)       AS click_pv,
            COUNT(CASE WHEN behavior_type = 2 THEN 1 END)       AS fav_pv,
            COUNT(CASE WHEN behavior_type = 3 THEN 1 END)       AS cart_pv,
            COUNT(CASE WHEN behavior_type = 4 THEN 1 END)       AS buy_pv,
            MAX(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END)  AS has_click,
            MAX(CASE WHEN behavior_type = 2 THEN 1 ELSE 0 END)  AS has_fav,
            MAX(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END)  AS has_cart,
            MAX(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END)  AS has_buy,
            MIN(hr)                                             AS first_hr,
            MAX(hr)                                             AS last_hr,
            COUNT(DISTINCT hr)                                  AS active_hours
        FROM dwd_user_behavior
        GROUP BY user_id, dt
    """)
    n1 = con.execute("SELECT COUNT(*) FROM dws_user_daily").fetchone()[0]
    print(f"  dws_user_daily: {n1:,} 行（用户 × 日）")

    print("\n  抽样 5 行：")
    show(con.execute(
        "SELECT * FROM dws_user_daily ORDER BY dt, user_id LIMIT 5").fetchdf())

    # ==========================================================
    # 【第 2 步】用户画像表
    # ==========================================================
    section("【第 2 步】dws_user_profile —— 用户画像表")
    print("  粒度：一个用户 一行\n")

    con.execute("DROP TABLE IF EXISTS dws_user_profile")
    con.execute("""
        CREATE TABLE dws_user_profile AS
        SELECT
            user_id,
            MIN(dt)                                            AS first_dt,
            MAX(dt)                                            AS last_dt,
            COUNT(DISTINCT dt)                                 AS active_days,
            SUM(pv)                                            AS total_pv,
            SUM(click_pv)                                      AS total_click_pv,
            SUM(fav_pv)                                        AS total_fav_pv,
            SUM(cart_pv)                                       AS total_cart_pv,
            SUM(buy_pv)                                        AS total_buy_pv,
            SUM(active_hours)                                  AS total_active_hours,
            ROUND(SUM(pv) * 1.0 / COUNT(DISTINCT dt), 2)       AS pv_per_active_day
        FROM dws_user_daily
        GROUP BY user_id
    """)
    n2 = con.execute("SELECT COUNT(*) FROM dws_user_profile").fetchone()[0]
    print(f"  dws_user_profile: {n2:,} 行（用户）")

    print("\n  关键分布：")
    show(con.execute("""
        SELECT
            COUNT(*)                                          AS 用户总数,
            ROUND(AVG(active_days), 2)                        AS 人均活跃天数,
            MEDIAN(active_days)                               AS 活跃天数中位数,
            ROUND(AVG(total_pv), 1)                           AS 人均行为数,
            ROUND(AVG(total_buy_pv), 2)                       AS 人均购买次数,
            SUM(CASE WHEN total_buy_pv = 0 THEN 1 ELSE 0 END) AS 从未购买用户数
        FROM dws_user_profile
    """).fetchdf())

    # ==========================================================
    # 【第 3 步】转化漏斗（修正版：双入口结构）
    # ==========================================================
    section("【第 3 步】ads_funnel —— 转化漏斗（修正版）")
    print("  修正说明：上一版用「加购→支付」做分母，得出 103% 的转化率。")
    print("            原因是【支付的人不都经过加购】，不能做单链漏斗。")
    print("            本版改为【双入口结构】：支付用户按实际路径拆分。\n")

    con.execute("DROP TABLE IF EXISTS ads_funnel")
    con.execute("""
        CREATE TABLE ads_funnel AS
        WITH u AS (
            SELECT user_id,
                   MAX(has_click) AS c,
                   MAX(has_fav)   AS f,
                   MAX(has_cart)  AS t,
                   MAX(has_buy)   AS b
            FROM dws_user_daily GROUP BY user_id
        ),
        agg AS (
            SELECT COUNT(*) AS uv,
                   SUM(c) AS click_uv,
                   SUM(t) AS cart_uv,
                   SUM(f) AS fav_uv,
                   SUM(b) AS buy_uv
            FROM u
        )
        SELECT '1-活跃用户' AS 层级, uv AS 用户数,
               NULL::DOUBLE AS 占活跃用户百分比,
               '全部分析基数' AS 说明 FROM agg
        UNION ALL
        SELECT '2-有点击', click_uv,
               ROUND(click_uv * 100.0 / NULLIF(uv, 0), 2),
               '浏览行为' FROM agg
        UNION ALL
        SELECT '3-有加购', cart_uv,
               ROUND(cart_uv * 100.0 / NULLIF(uv, 0), 2),
               '高意向行为' FROM agg
        UNION ALL
        SELECT '3b-有收藏', fav_uv,
               ROUND(fav_uv * 100.0 / NULLIF(uv, 0), 2),
               '弱意向行为' FROM agg
        UNION ALL
        SELECT '4-有支付', buy_uv,
               ROUND(buy_uv * 100.0 / NULLIF(uv, 0), 2),
               '成交（北极星）' FROM agg
    """)

    print("  各环节用户数（相对活跃用户，不做单链转化率）：")
    show(con.execute("SELECT 层级, 用户数, 占活跃用户百分比, 说明 "
                     "FROM ads_funnel").fetchdf())

    # ---- 路径交叉分析 ----
    print("\n  支付用户的路径拆解（互斥分类，这才是正确的漏斗）：")
    con.execute("DROP TABLE IF EXISTS ads_path_breakdown")
    con.execute("""
        CREATE TABLE ads_path_breakdown AS
        WITH u AS (
            SELECT user_id,
                   MAX(has_fav)  AS f,
                   MAX(has_cart) AS t,
                   MAX(has_buy)  AS b
            FROM dws_user_daily GROUP BY user_id
        ),
        graded AS (
            SELECT CASE
                WHEN b = 0 THEN 'A-未支付'
                WHEN b = 1 AND t = 1 AND f = 1 THEN 'B-加购+收藏后支付'
                WHEN b = 1 AND t = 1 AND f = 0 THEN 'C-仅加购后支付'
                WHEN b = 1 AND t = 0 AND f = 1 THEN 'D-仅收藏后支付'
                ELSE                                  'E-直接支付（未加购未收藏）'
            END AS path
            FROM u
        )
        SELECT path AS 路径, COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占全部用户百分比
        FROM graded GROUP BY path ORDER BY path
    """)
    show(con.execute("""
        SELECT 路径, 用户数, 占全部用户百分比,
               ROUND(用户数 * 100.0 /
                     (SELECT SUM(用户数) FROM ads_path_breakdown
                      WHERE 路径 <> 'A-未支付'), 2) AS 占支付用户百分比
        FROM ads_path_breakdown ORDER BY 路径
    """).fetchdf())

    # ---- PV 口径对照 + 摩擦成本 ----
    print("\n  摩擦成本指标（PV 口径的正确用法）：")
    print("     说明：不看'事件转化率'，看'完成一次成交需要多少次点击'\n")
    show(con.execute("""
        WITH s AS (
            SELECT
                COUNT(CASE WHEN behavior_type = 1 THEN 1 END) AS click_pv,
                COUNT(CASE WHEN behavior_type = 3 THEN 1 END) AS cart_pv,
                COUNT(CASE WHEN behavior_type = 4 THEN 1 END) AS buy_pv
            FROM dwd_user_behavior
        )
        SELECT click_pv AS 点击事件数, cart_pv AS 加购事件数, buy_pv AS 支付事件数,
               ROUND(click_pv * 1.0 / NULLIF(buy_pv, 0), 1) AS 每次支付需点击次数,
               ROUND(click_pv * 1.0 / NULLIF(cart_pv, 0), 1) AS 每次加购需点击次数
        FROM s
    """).fetchdf())

    # ==========================================================
    # 【第 4 步】同期群留存
    # ==========================================================
    section("【第 4 步】ads_retention —— 同期群留存")
    print("  口径：按用户【首次活跃日】分组，追踪之后第 N 天的回访情况\n")

    con.execute("DROP TABLE IF EXISTS ads_retention")
    con.execute("""
        CREATE TABLE ads_retention AS
        WITH first_seen AS (
            SELECT user_id, MIN(dt) AS cohort_dt
            FROM dws_user_daily GROUP BY user_id
        ),
        joined AS (
            SELECT d.user_id, d.dt, f.cohort_dt,
                   DATE_DIFF('day', f.cohort_dt, d.dt) AS day_n,
                   d.has_buy
            FROM dws_user_daily d
            JOIN first_seen f ON d.user_id = f.user_id
        )
        SELECT cohort_dt, day_n,
               COUNT(DISTINCT user_id)                                AS active_users,
               COUNT(DISTINCT CASE WHEN has_buy = 1 THEN user_id END) AS buyer_users
        FROM joined
        GROUP BY cohort_dt, day_n
        ORDER BY cohort_dt, day_n
    """)
    n4 = con.execute("SELECT COUNT(*) FROM ads_retention").fetchone()[0]
    print(f"  ads_retention: {n4:,} 行")

    print("\n  首批用户（2014-11-18）的留存曲线（前 14 天）：")
    show(con.execute("""
        WITH base AS (SELECT * FROM ads_retention WHERE cohort_dt = '2014-11-18')
        SELECT 第N天, 活跃用户数, 留存率百分比, 付费用户数
        FROM (
            SELECT day_n AS 第N天, active_users AS 活跃用户数,
                   ROUND(active_users * 100.0 /
                         (SELECT active_users FROM base WHERE day_n = 0), 2) AS 留存率百分比,
                   buyer_users AS 付费用户数
            FROM base WHERE day_n <= 14
        ) ORDER BY 第N天
    """).fetchdf())

    print("\n  各同期群 D1/D7 留存（注意队列规模，小样本不可解读）：")
    show(con.execute("""
        WITH d0 AS (SELECT cohort_dt, active_users AS base
                    FROM ads_retention WHERE day_n = 0)
        SELECT r.cohort_dt AS 首次活跃日, d0.base AS 队列规模,
               CASE WHEN d0.base < 100 THEN '样本过小-不解读'
                    WHEN d0.base < 500 THEN '样本偏小-谨慎'
                    ELSE '可解读' END AS 可信度,
               ROUND(MAX(CASE WHEN r.day_n = 1 THEN r.active_users END) * 100.0 / d0.base, 2) AS D1留存率,
               ROUND(MAX(CASE WHEN r.day_n = 7 THEN r.active_users END) * 100.0 / d0.base, 2) AS D7留存率
        FROM ads_retention r JOIN d0 ON r.cohort_dt = d0.cohort_dt
        GROUP BY r.cohort_dt, d0.base ORDER BY r.cohort_dt
    """).fetchdf(), max_rows=40)

    # ==========================================================
    # 【第 5 步】RFM 分层
    # ==========================================================
    section("【第 5 步】ads_rfm —— RFM 用户分层")
    print("  R = 距末次日期的天数  F = 活跃天数  M = 购买事件数")
    print("  [注意] 无价格字段，M 由「金额」降级为「购买次数」\n")

    con.execute("DROP TABLE IF EXISTS ads_rfm")
    con.execute(f"""
        CREATE TABLE ads_rfm AS
        WITH base AS (
            SELECT user_id,
                   DATE_DIFF('day', max_dt, DATE '{config.DATA_END_DATE}') AS recency,
                   active_days,
                   total_buy_pv AS buy_pv
            FROM (
                SELECT user_id, MAX(dt) AS max_dt, COUNT(DISTINCT dt) AS active_days,
                       SUM(buy_pv) AS total_buy_pv
                FROM dws_user_daily GROUP BY user_id
            )
        ),
        scored AS (
            SELECT user_id, recency, active_days, buy_pv,
                   NTILE(4) OVER (ORDER BY recency DESC)    AS r_score,
                   NTILE(4) OVER (ORDER BY active_days ASC) AS f_score,
                   NTILE(4) OVER (ORDER BY buy_pv ASC)      AS m_score
            FROM base
        )
        SELECT
            user_id, recency, active_days, buy_pv,
            r_score, f_score, m_score,
            (r_score + f_score + m_score) AS rfm_total,
            CASE
                WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN '重要价值用户'
                WHEN r_score >= 3 AND f_score >= 3 AND m_score <  3 THEN '重要保持用户'
                WHEN r_score >= 3 AND f_score <  3 AND m_score >= 3 THEN '重要发展用户'
                WHEN r_score >= 3 AND f_score <  3 AND m_score <  3 THEN '新客'
                WHEN r_score <  3 AND f_score >= 3 AND m_score >= 3 THEN '重要挽留用户'
                WHEN r_score <  3 AND f_score >= 3 AND m_score <  3 THEN '一般保持用户'
                WHEN r_score <  3 AND f_score <  3 AND m_score >= 3 THEN '一般发展用户'
                ELSE '流失用户'
            END AS rfm_segment
        FROM scored
    """)

    print("  RFM 分层结果：")
    show(con.execute("""
        SELECT rfm_segment AS 用户分层, COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
               ROUND(AVG(recency), 1)     AS 平均最近间隔天数,
               ROUND(AVG(active_days), 1) AS 平均活跃天数,
               ROUND(AVG(buy_pv), 1)      AS 平均购买次数
        FROM ads_rfm GROUP BY 1 ORDER BY 用户数 DESC
    """).fetchdf(), max_rows=20)

    print("\n  [WARN] 分层合理性检查（看各层是否真有差异）：")
    show(con.execute("""
        SELECT MIN(recency) AS R最小, MAX(recency) AS R最大,
               MIN(active_days) AS F最小, MAX(active_days) AS F最大,
               MIN(buy_pv) AS M最小, MAX(buy_pv) AS M最大
        FROM ads_rfm
    """).fetchdf())

    # ==========================================================
    # 【第 6 步】商品级质量信号
    # ==========================================================
    section("【第 6 步】ads_item_quality —— 商品级质量信号")
    print("     若占比高 → 该疑点成立，加购率会低估真实购买意向\n")

    con.execute("DROP TABLE IF EXISTS ads_item_quality")
    con.execute("""
        CREATE TABLE ads_item_quality AS
        WITH per_item AS (
            SELECT item_id,
                   COUNT(CASE WHEN behavior_type = 1 THEN 1 END) AS clicks,
                   COUNT(CASE WHEN behavior_type = 2 THEN 1 END) AS favs,
                   COUNT(CASE WHEN behavior_type = 3 THEN 1 END) AS carts,
                   COUNT(CASE WHEN behavior_type = 4 THEN 1 END) AS buys,
                   COUNT(DISTINCT user_id)                        AS users,
                   SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END)
                     + SUM(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END) AS _c
            FROM dwd_user_behavior
            GROUP BY item_id
        )
        SELECT
            item_id, users, clicks, favs, carts, buys,
            ROUND(carts * 100.0 / NULLIF(clicks, 0), 2) AS cart_rate,
            ROUND(buys  * 100.0 / NULLIF(clicks, 0), 2) AS buy_rate,
            ROUND(clicks * 1.0 / NULLIF(users, 0), 2)   AS clicks_per_user
        FROM per_item
        WHERE clicks >= 50
    """)
    n6 = con.execute("SELECT COUNT(*) FROM ads_item_quality").fetchone()[0]
    print(f"  ads_item_quality: {n6:,} 个商品（点击 >= 50 次）")

    print("\n  高意向商品 Top 10（点击→加购率最高，要求点击 >= 200）：")
    show(con.execute("""
        SELECT item_id AS 商品ID, users AS 触达用户, clicks AS 点击, carts AS 加购, buys AS 支付,
               cart_rate AS 点击转加购率, buy_rate AS 点击转支付率
        FROM ads_item_quality WHERE clicks >= 200
        ORDER BY cart_rate DESC LIMIT 10
    """).fetchdf())

    print("\n  [!] 需优化商品 Top 10（点击多但加购率极低，点击 >= 500）：")
    show(con.execute("""
        SELECT item_id AS 商品ID, users AS 触达用户, clicks AS 点击, carts AS 加购, buys AS 支付,
               cart_rate AS 点击转加购率, buy_rate AS 点击转支付率
        FROM ads_item_quality WHERE clicks >= 500
        ORDER BY cart_rate ASC LIMIT 10
    """).fetchdf())

    print("\n  商品级点击转加购率的分位数分布（看整体形态）：")
    show(con.execute("""
        SELECT COUNT(*) AS 商品数,
               ROUND(QUANTILE_CONT(cart_rate, 0.25), 2) AS p25,
               ROUND(QUANTILE_CONT(cart_rate, 0.50), 2) AS p50中位数,
               ROUND(QUANTILE_CONT(cart_rate, 0.75), 2) AS p75,
               ROUND(QUANTILE_CONT(cart_rate, 0.90), 2) AS p90,
               ROUND(MAX(cart_rate), 2) AS 最大
        FROM ads_item_quality
    """).fetchdf())

    # ==========================================================
    # 【第 7 步】同用户·同商品重复点击（剔除个人习惯干扰）
    # ==========================================================
    section("【第 7 步】重复点击分析 —— 剔除个人浏览习惯的干扰")
    print("  方法：把点击拆成两类")
    print("        ① 同用户·同商品重复点击  → 反映【商品】的吸引力")
    print("        ② 同用户·跨商品点击      → 反映【用户】的浏览习惯\n")

    con.execute("DROP TABLE IF EXISTS ads_repeat_click")
    con.execute("""
        CREATE TABLE ads_repeat_click AS
        WITH clicks AS (
            SELECT user_id, item_id, COUNT(*) AS click_cnt
            FROM dwd_user_behavior
            WHERE behavior_type = 1
            GROUP BY user_id, item_id
        )
        SELECT
            item_id,
            COUNT(*)                                     AS 触达用户商品对,
            COUNT(DISTINCT user_id)                      AS 触达用户数,
            SUM(click_cnt)                               AS 点击总数,
            SUM(CASE WHEN click_cnt > 1 THEN click_cnt - 1 ELSE 0 END) AS 重复点击次数,
            ROUND(SUM(CASE WHEN click_cnt > 1 THEN click_cnt - 1 ELSE 0 END)
                  * 100.0 / NULLIF(SUM(click_cnt), 0), 2)              AS 重复点击占比,
            ROUND(AVG(click_cnt), 2)                     AS 人均点击次数
        FROM clicks
        GROUP BY item_id
        HAVING SUM(click_cnt) >= 200
    """)
    n7 = con.execute("SELECT COUNT(*) FROM ads_repeat_click").fetchone()[0]
    print(f"  ads_repeat_click: {n7:,} 个商品")

    print("\n  用户重复回访最多的商品 Top 10（剔除个人习惯后的强意向信号）：")
    show(con.execute("""
        SELECT item_id AS 商品ID, 触达用户数, 点击总数, 重复点击次数,
               重复点击占比, 人均点击次数
        FROM ads_repeat_click ORDER BY 重复点击占比 DESC LIMIT 10
    """).fetchdf())

    print("\n  整体重复点击占比分布：")
    show(con.execute("""
        SELECT COUNT(*) AS 商品数,
               ROUND(AVG(重复点击占比), 2) AS 平均重复占比,
               ROUND(QUANTILE_CONT(重复点击占比, 0.50), 2) AS p50中位数,
               ROUND(QUANTILE_CONT(重复点击占比, 0.90), 2) AS p90
        FROM ads_repeat_click
    """).fetchdf())

    # ==========================================================
    # 【第 8 步】导出
    # ==========================================================
    section("【第 8 步】导出结果")

    import pandas as pd
    for t in ["dws_user_daily", "dws_user_profile", "ads_funnel",
              "ads_path_breakdown", "ads_retention", "ads_rfm",
              "ads_item_quality", "ads_repeat_click"]:
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<20} {cnt:>10,} 行")

    xlsx_path = config.TABLE_DIR / "P2_指标结果.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("SELECT 层级, 用户数, 占活跃用户百分比, 说明 "
                        "FROM ads_funnel").fetchdf().to_excel(
                writer, sheet_name="漏斗", index=False)
            con.execute("SELECT * FROM ads_path_breakdown").fetchdf().to_excel(
                writer, sheet_name="路径拆解", index=False)
            con.execute("""
                SELECT rfm_segment AS 用户分层, COUNT(*) AS 用户数,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
                       ROUND(AVG(recency),1) AS 平均最近间隔天数,
                       ROUND(AVG(active_days),1) AS 平均活跃天数,
                       ROUND(AVG(buy_pv),1) AS 平均购买次数
                FROM ads_rfm GROUP BY 1 ORDER BY 用户数 DESC
            """).fetchdf().to_excel(writer, sheet_name="RFM分层", index=False)
            con.execute("""
                SELECT 商品ID, 触达用户数, 点击总数, 重复点击次数, 重复点击占比, 人均点击次数
                FROM (SELECT item_id AS 商品ID, 触达用户数, 点击总数,
                             重复点击次数, 重复点击占比, 人均点击次数
                      FROM ads_repeat_click ORDER BY 重复点击占比 DESC LIMIT 100)
            """).fetchdf().to_excel(writer, sheet_name="高意向商品", index=False)
            con.execute("""
                SELECT 商品ID, 触达用户, 点击, 加购, 支付, 点击转加购率, 点击转支付率
                FROM (SELECT item_id AS 商品ID, users AS 触达用户, clicks AS 点击,
                             carts AS 加购, buys AS 支付, cart_rate AS 点击转加购率,
                             buy_rate AS 点击转支付率
                      FROM ads_item_quality WHERE clicks >= 200
                      ORDER BY cart_rate DESC LIMIT 100)
            """).fetchdf().to_excel(writer, sheet_name="高意向商品_转化率", index=False)
        print(f"\n  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal

    section("P2 执行完成")
    print(f"  总耗时  : {time.time() - t0:.1f} 秒")
    print(f"  运行日志: {log_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
