# -*- coding: utf-8 -*-
"""
P3 step 1: refine user segmentation, then attribute the conversion gap.

The largest segment (high-frequency low-conversion, 30.77%) is nearly as
active as the core value segment (25.9 vs 27.9 active days) but buys 4.2x
less (7.9 vs 33.4 purchases). This script tests four hypotheses about why:

    H1  they browse with no focus        -> fragmented category browsing
    H2  they buy mainly during promos    -> promo-dependent timing
    H3  they add to cart but do not pay  -> intent without conversion
    H4  they rarely add to cart either   -> no genuine interest

Usage:
    python src/p3_attribution.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68


class Tee:
    """Mirror stdout AND stderr to one log file."""

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
    pd.set_option("display.width", 240)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))


def main():
    t0 = time.time()
    log_path = config.TABLE_DIR / "P3_运行日志.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(DIVIDER)
    print("  P3 用户流失归因分析")
    print(DIVIDER)

    con = duckdb.connect(str(config.DUCKDB_PATH))

    # ==========================================================
    # 【第 1 步】细分用户分层（修正 P2 的中频用户被吸收问题）
    # ==========================================================
    section("【第 1 步】ads_user_base —— 细分用户分层")
    print("  修正点：P2 的兜底规则把中等活跃度用户吸收进了'一般活跃用户'。")
    print("          本版按【活跃度 × 转化率】二维交叉，共 8 类。\n")

    con.execute("DROP TABLE IF EXISTS ads_user_base")
    con.execute(f"""
        CREATE TABLE ads_user_base AS
        WITH base AS (
            SELECT user_id,
                   DATE_DIFF('day', MAX(dt), DATE '{config.DATA_END_DATE}') AS recency,
                   COUNT(DISTINCT dt)  AS active_days,
                   SUM(pv)             AS total_pv,
                   SUM(click_pv)       AS total_click,
                   SUM(cart_pv)        AS total_cart,
                   SUM(fav_pv)         AS total_fav,
                   SUM(buy_pv)         AS total_buy
            FROM dws_user_daily GROUP BY user_id
        ),
        graded AS (
            SELECT *,
                   CASE WHEN recency <= 3  THEN '高'
                        WHEN recency <= 7  THEN '中'
                        WHEN recency <= 14 THEN '低'
                        ELSE '流失' END        AS r_grade,
                   CASE WHEN active_days > 20 THEN '高'
                        WHEN active_days >= 8 THEN '中'
                        ELSE '低' END          AS f_grade,
                   CASE WHEN total_buy > 15 THEN '高'
                        WHEN total_buy >= 6 THEN '中'
                        WHEN total_buy >= 1 THEN '低'
                        ELSE '无' END          AS m_grade,
                   ROUND(total_buy * 1.0
                         / NULLIF(active_days, 0), 3) AS buy_per_day
            FROM base
        )
        SELECT
            user_id, recency, active_days, total_pv, total_click,
            total_cart, total_fav, total_buy, buy_per_day,
            r_grade, f_grade, m_grade,
            CASE
                WHEN m_grade = '无'                            THEN '0-未购买用户'
                WHEN r_grade = '流失'                          THEN '7-已流失用户'
                WHEN r_grade = '低'                            THEN '6-流失预警用户'
                WHEN f_grade = '高' AND m_grade = '高'          THEN '1-核心价值用户'
                WHEN f_grade = '高'                            THEN '2-高频低转化用户'
                WHEN f_grade = '中' AND m_grade = '高'          THEN '3-中频高转化用户'
                WHEN f_grade = '中'                            THEN '4-中频低转化用户'
                WHEN f_grade = '低' AND m_grade = '高'          THEN '3-中频高转化用户'
                WHEN r_grade = '高'                            THEN '5-新客'
                ELSE                                            '4-中频低转化用户'
            END AS segment
        FROM graded
    """)

    print("  细分后的分层结果：")
    show(con.execute("""
        SELECT segment AS 用户分层, COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
               ROUND(AVG(recency), 1)     AS 平均最近间隔,
               ROUND(AVG(active_days), 1) AS 平均活跃天数,
               ROUND(AVG(total_buy), 1)   AS 平均购买次数,
               ROUND(AVG(buy_per_day), 3) AS 日均购买次数
        FROM ads_user_base GROUP BY 1 ORDER BY 1
    """).fetchdf(), max_rows=20)

    print("\n  各层的活跃度 / 转化率二维对照（看是否真的分化）：")
    show(con.execute("""
        SELECT segment AS 用户分层,
               ROUND(AVG(active_days), 1) AS 活跃天数,
               ROUND(AVG(total_click), 0) AS 平均点击,
               ROUND(AVG(total_cart), 1)  AS 平均加购,
               ROUND(AVG(total_buy), 1)   AS 平均购买,
               ROUND(AVG(total_cart) * 100.0
                     / NULLIF(AVG(total_click), 0), 2) AS 加购率百分比,
               ROUND(AVG(total_buy) * 100.0
                     / NULLIF(AVG(total_cart), 0), 2)  AS 加购转支付率百分比
        FROM ads_user_base GROUP BY 1 ORDER BY 1
    """).fetchdf(), max_rows=20)

    # ==========================================================
    # 【第 2 步】H1：浏览是否有聚焦（品类集中度）
    # ==========================================================
    section("【第 2 步】H1 检验 —— 他们浏览商品是有聚焦还是漫无目的？")
    print("  指标：人均浏览品类数、单品类最大占比（HHI 思路）")
    print("        品类数少 + 单品类占比高 = 有明确偏好\n")

    con.execute("DROP TABLE IF EXISTS tmp_user_category")
    con.execute("""
        CREATE TABLE tmp_user_category AS
        SELECT b.segment, d.user_id,
               COUNT(DISTINCT d.item_category) AS cat_cnt,
               COUNT(*)                        AS events
        FROM dwd_user_behavior d
        JOIN ads_user_base b ON d.user_id = b.user_id
        GROUP BY b.segment, d.user_id
    """)

    show(con.execute("""
        SELECT segment AS 用户分层,
               COUNT(*) AS 用户数,
               ROUND(AVG(cat_cnt), 1) AS 人均浏览品类数,
               ROUND(MEDIAN(cat_cnt), 0) AS 品类数中位数,
               ROUND(AVG(events), 0) AS 人均行为数
        FROM tmp_user_category GROUP BY 1 ORDER BY 1
    """).fetchdf(), max_rows=20)

    # ==========================================================
    # 【第 3 步】H2：购买是否集中在大促
    # ==========================================================
    section("【第 3 步】H2 检验 —— 他们的购买是否集中在大促？")
    print("  指标：大促窗口（12-10~12-12）的购买事件占比\n")

    con.execute("DROP TABLE IF EXISTS tmp_promo")
    con.execute("""
        CREATE TABLE tmp_promo AS
        SELECT b.segment,
               COUNT(*)                                        AS buy_events_total,
               COUNT(CASE WHEN p.is_promo = 1 THEN 1 END)      AS buy_events_promo,
               COUNT(DISTINCT d.user_id)                       AS buyers,
               COUNT(DISTINCT CASE WHEN p.is_promo = 1 THEN d.user_id END) AS promo_buyers
        FROM dwd_user_behavior d
        JOIN ads_user_base b ON d.user_id = b.user_id
        JOIN dim_date p ON d.dt = p.dt
        WHERE d.behavior_type = 4
        GROUP BY b.segment
    """)

    show(con.execute("""
        SELECT segment AS 用户分层, buyers AS 购买人数,
               buy_events_total AS 总购买事件,
               buy_events_promo AS 大促期购买事件,
               ROUND(buy_events_promo * 100.0
                     / NULLIF(buy_events_total, 0), 2) AS 大促购买占比百分比,
               ROUND(buy_events_total * 1.0
                     / NULLIF(buyers, 0), 2) AS 人均购买次数
        FROM tmp_promo ORDER BY 1
    """).fetchdf(), max_rows=20)

    print("\n  对照：大促窗口只占全周期 3/31 = 9.68% 的天数")
    print("        → 如果某人群大促购买占比显著高于 9.68%，说明是价格敏感型\n")

    # ==========================================================
    # 【第 4 步】H3/H4：加购后是否支付
    # ==========================================================
    section("【第 4 步】H3/H4 检验 —— 有意向却不买，还是根本没意向？")
    print("  核心指标：加购→支付转化率")
    print("           高 = 有意向但犹豫（H3）  低 = 没真意向（H4）\n")

    con.execute("DROP TABLE IF EXISTS ads_segment_behavior")
    con.execute("""
        CREATE TABLE ads_segment_behavior AS
        SELECT
            segment,
            COUNT(*)                                   AS users,
            ROUND(AVG(active_days), 1)                 AS avg_active_days,
            ROUND(AVG(total_click), 0)                 AS avg_click,
            ROUND(AVG(total_cart), 1)                  AS avg_cart,
            ROUND(AVG(total_fav), 1)                   AS avg_fav,
            ROUND(AVG(total_buy), 1)                   AS avg_buy,
            ROUND(SUM(total_cart) * 100.0
                  / NULLIF(SUM(total_click), 0), 2)    AS 加购率百分比,
            ROUND(SUM(total_buy) * 100.0
                  / NULLIF(SUM(total_cart), 0), 2)     AS 加购转支付率百分比,
            ROUND(SUM(total_fav) * 100.0
                  / NULLIF(SUM(total_click), 0), 2)    AS 收藏率百分比
        FROM ads_user_base
        GROUP BY segment ORDER BY segment
    """)

    show(con.execute("SELECT * FROM ads_segment_behavior").fetchdf(), max_rows=20)

    print("\n  ※ 重点看【加购转支付率】这一列：")
    print("     若高频低转化用户 < 核心价值用户 → 有意向但不成交（H3）")
    print("     若两者接近但加购率都很低       → 根本没意向（H4）\n")

    # ==========================================================
    # 【第 5 步】核心对比：高频低转化 vs 核心价值
    # ==========================================================
    section("【第 5 步】核心对比 —— 高频低转化 vs 核心价值用户")
    print("  目的：找出两个人群（活跃度相近、购买差 4 倍）的关键差异\n")

    show(con.execute("""
        SELECT
            segment AS 人群,
            users AS 人数,
            avg_active_days AS 平均活跃天数,
            avg_click  AS 平均点击,
            avg_cart   AS 平均加购,
            avg_fav    AS 平均收藏,
            avg_buy    AS 平均支付,
            加购率百分比,
            加购转支付率百分比
        FROM ads_segment_behavior
        WHERE segment IN ('1-核心价值用户', '2-高频低转化用户', '4-中频低转化用户')
        ORDER BY segment
    """).fetchdf())

    print("\n  关键差值计算：")
    show(con.execute("""
        WITH core AS (
            SELECT * FROM ads_segment_behavior WHERE segment = '1-核心价值用户'
        ),
        hf AS (
            SELECT * FROM ads_segment_behavior WHERE segment = '2-高频低转化用户'
        )
        SELECT
            '活跃天数' AS 对比维度,
            core.avg_active_days AS 核心价值用户,
            hf.avg_active_days   AS 高频低转化用户,
            ROUND(hf.avg_active_days * 1.0 / core.avg_active_days, 2) AS 倍数
        FROM core, hf
        UNION ALL SELECT '平均点击', core.avg_click, hf.avg_click,
            ROUND(hf.avg_click * 1.0 / core.avg_click, 2) FROM core, hf
        UNION ALL SELECT '平均加购', core.avg_cart, hf.avg_cart,
            ROUND(hf.avg_cart * 1.0 / core.avg_cart, 2) FROM core, hf
        UNION ALL SELECT '平均支付', core.avg_buy, hf.avg_buy,
            ROUND(hf.avg_buy * 1.0 / core.avg_buy, 2) FROM core, hf
        UNION ALL SELECT '加购转支付率', core.加购转支付率百分比, hf.加购转支付率百分比,
            ROUND(hf.加购转支付率百分比 * 1.0 / core.加购转支付率百分比, 2) FROM core, hf
    """).fetchdf())

    # ==========================================================
    # 【第 6 步】未购买用户专项
    # ==========================================================
    section("【第 6 步】未购买用户专项 —— 1,114 人为什么一次都没买？")
    print("  检查：他们到底有没有产生过购买意向（加购/收藏）\n")

    show(con.execute("""
        SELECT
            COUNT(*)                                   AS 未购买用户数,
            ROUND(AVG(active_days), 1)                 AS 平均活跃天数,
            ROUND(AVG(total_click), 0)                 AS 平均点击,
            ROUND(AVG(total_cart), 1)                  AS 平均加购,
            ROUND(AVG(total_fav), 1)                   AS 平均收藏,
            SUM(CASE WHEN total_cart > 0 THEN 1 ELSE 0 END) AS 加购过的用户数,
            SUM(CASE WHEN total_cart = 0 AND total_fav = 0
                     THEN 1 ELSE 0 END)                AS 从未加购也未收藏的用户数
        FROM ads_user_base WHERE segment = '0-未购买用户'
    """).fetchdf())

    print("\n  ※ 解读：")
    print("     若'从未加购也未收藏'占比高 → 他们只是来逛，没有购买意图")
    print("     若'加购过的用户'占比高     → 有意向但最终没成交，是转化问题\n")

    # ==========================================================
    # 【第 7 步】导出
    # ==========================================================
    section("【第 7 步】导出结果")
    import pandas as pd
    xlsx_path = config.TABLE_DIR / "P3_归因结果.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("""
                SELECT segment AS 用户分层, COUNT(*) AS 用户数,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
                       ROUND(AVG(active_days),1) AS 平均活跃天数,
                       ROUND(AVG(total_buy),1) AS 平均购买次数
                FROM ads_user_base GROUP BY 1 ORDER BY 1
            """).fetchdf().to_excel(writer, sheet_name="细分分层", index=False)
            con.execute("SELECT * FROM ads_segment_behavior").fetchdf().to_excel(
                writer, sheet_name="人群行为对比", index=False)
            con.execute("SELECT * FROM tmp_promo ORDER BY 1").fetchdf().to_excel(
                writer, sheet_name="大促依赖度", index=False)
        print(f"  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal

    section("P3 归因分析完成")
    print(f"  耗时  : {time.time() - t0:.1f} 秒")
    print(f"  运行日志: {log_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
