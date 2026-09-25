# -*- coding: utf-8 -*-
"""
P2-fix: friction cost correction and threshold-based user segmentation.

Two corrections on top of the first p2_metrics run:

  1. Friction cost must use DISTINCT (user, item) clicks, not raw clicks.
     Repeat clicks account for ~65% of all clicks, which inflates the
     raw ratio by roughly 3x.

  2. RFM segmentation with NTILE(4) is not usable on a 31-day window:
     the R quartile boundary lands at ~4 days, labelling active users as
     churned. Fixed business thresholds are applied instead.

Usage:
    python src/p2_fix_segmentation.py
"""
import sys
import time
import traceback
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68


class Tee:
    """Mirror stdout AND stderr to one log file.

    stderr is redirected to the same file handle as stdout so that
    tracebacks are captured too, instead of only appearing in red in
    the PyCharm console.
    """

    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


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
    log_path = config.TABLE_DIR / "P2fix_运行日志.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee          # 关键：让报错也写进日志

    print(DIVIDER)
    print("  P2 修正：摩擦成本口径 + 用户分层方案")
    print(DIVIDER)

    con = duckdb.connect(str(config.DUCKDB_PATH))

    # ==========================================================
    # 【修正 1】摩擦成本：区分原始点击与有效点击
    # ==========================================================
    section("【修正 1】摩擦成本 —— 原始点击 vs 有效点击")
    print("  背景：重复点击占比约 65%，用原始点击数算摩擦成本会虚高约 3 倍。")
    print("  定义：有效点击 = 按 (用户, 商品) 去重后的点击数")
    print("        即「用户实际探索过多少个不同的商品」\n")

    con.execute("DROP TABLE IF EXISTS ads_friction_cost")
    con.execute("""
        CREATE TABLE ads_friction_cost AS
        WITH raw AS (
            SELECT
                COUNT(CASE WHEN behavior_type = 1 THEN 1 END) AS raw_clicks,
                COUNT(CASE WHEN behavior_type = 3 THEN 1 END) AS cart_events,
                COUNT(CASE WHEN behavior_type = 4 THEN 1 END) AS buy_events
            FROM dwd_user_behavior
        ),
        distinct_clicks AS (
            SELECT COUNT(*) AS effective_clicks FROM (
                SELECT DISTINCT user_id, item_id
                FROM dwd_user_behavior WHERE behavior_type = 1
            )
        )
        SELECT
            r.raw_clicks, d.effective_clicks, r.cart_events, r.buy_events,
            ROUND(r.raw_clicks * 1.0 / NULLIF(r.buy_events, 0), 1)           AS 原始点击_每次支付,
            ROUND(d.effective_clicks * 1.0 / NULLIF(r.buy_events, 0), 1)     AS 有效点击_每次支付,
            ROUND(r.raw_clicks * 1.0 / NULLIF(r.cart_events, 0), 1)          AS 原始点击_每次加购,
            ROUND(d.effective_clicks * 1.0 / NULLIF(r.cart_events, 0), 1)    AS 有效点击_每次加购,
            ROUND((1 - d.effective_clicks * 1.0 / NULLIF(r.raw_clicks, 0)) * 100, 2) AS 重复点击占比百分比
        FROM raw r, distinct_clicks d
    """)

    show(con.execute("""
        SELECT 原始点击_每次支付, 有效点击_每次支付,
               原始点击_每次加购, 有效点击_每次加购,
               重复点击占比百分比
        FROM ads_friction_cost
    """).fetchdf())

    print("\n  → 两个口径都要保留：")
    print("     原始点击口径 = 用户实际的操作成本（含重复浏览）")
    print("     有效点击口径 = 用户需要探索多少个不同商品，才成交一次\n")

    # ==========================================================
    # 【修正 2】用户分层：固定业务阈值（替代 NTILE）
    # ==========================================================
    section("【修正 2】ads_user_segment —— 固定阈值用户分层")
    print("  背景：NTILE(4) 的 R 分位边界落在约 4 天，把活跃用户标为流失用户。")
    print("        根因：NTILE 只保证组数均匀，不保证分组有业务意义。")
    print("  方案：改用固定业务阈值，依据是本数据的实际分布。\n")

    print("  阈值设定依据（先看数据分布）：")
    show(con.execute(f"""
        WITH base AS (
            SELECT user_id,
                   DATE_DIFF('day', MAX(dt), DATE '{config.DATA_END_DATE}') AS recency,
                   COUNT(DISTINCT dt) AS active_days,
                   SUM(buy_pv)        AS buy_pv
            FROM dws_user_daily GROUP BY user_id
        )
        SELECT
            ROUND(QUANTILE_CONT(recency, 0.25),0) AS R_p25,
            ROUND(QUANTILE_CONT(recency, 0.50),0) AS R_p50,
            ROUND(QUANTILE_CONT(recency, 0.75),0) AS R_p75,
            ROUND(QUANTILE_CONT(active_days, 0.25),0) AS F_p25,
            ROUND(QUANTILE_CONT(active_days, 0.50),0) AS F_p50,
            ROUND(QUANTILE_CONT(active_days, 0.75),0) AS F_p75,
            ROUND(QUANTILE_CONT(buy_pv, 0.25),0) AS M_p25,
            ROUND(QUANTILE_CONT(buy_pv, 0.50),0) AS M_p50,
            ROUND(QUANTILE_CONT(buy_pv, 0.75),0) AS M_p75
        FROM base
    """).fetchdf())

    print("\n  采用阈值：")
    print("     R: 0-3 高 | 4-7 中 | 8-14 低 | >14 流失")
    print("     F: >20 高 | 8-20 中 | <=7 低")
    print("     M: >15 高 | 6-15 中 | 1-5 低 | 0 无购买\n")

    con.execute("DROP TABLE IF EXISTS ads_user_segment")
    con.execute(f"""
        CREATE TABLE ads_user_segment AS
        WITH base AS (
            SELECT user_id,
                   DATE_DIFF('day', MAX(dt), DATE '{config.DATA_END_DATE}') AS recency,
                   COUNT(DISTINCT dt) AS active_days,
                   SUM(buy_pv)        AS buy_pv
            FROM dws_user_daily GROUP BY user_id
        ),
        graded AS (
            SELECT user_id, recency, active_days, buy_pv,
                   CASE WHEN recency <= 3  THEN '高'
                        WHEN recency <= 7  THEN '中'
                        WHEN recency <= 14 THEN '低'
                        ELSE '流失' END AS r_grade,
                   CASE WHEN active_days > 20 THEN '高'
                        WHEN active_days >= 8 THEN '中'
                        ELSE '低' END AS f_grade,
                   CASE WHEN buy_pv > 15 THEN '高'
                        WHEN buy_pv >= 6 THEN '中'
                        WHEN buy_pv >= 1 THEN '低'
                        ELSE '无' END AS m_grade
            FROM base
        )
        SELECT
            user_id, recency, active_days, buy_pv,
            r_grade, f_grade, m_grade,
            CASE
                WHEN m_grade = '无'                   THEN '未购买用户'
                WHEN r_grade = '流失'                 THEN '已流失用户'
                WHEN r_grade = '低'                   THEN '流失预警用户'
                WHEN r_grade IN ('高','中') AND f_grade = '高' AND m_grade = '高' THEN '核心价值用户'
                WHEN r_grade IN ('高','中') AND f_grade = '高'                    THEN '高频低转化用户'
                WHEN r_grade IN ('高','中') AND m_grade = '高'                    THEN '低频高转化用户'
                WHEN r_grade = '高' AND active_days <= 12                        THEN '新客'
                ELSE '一般活跃用户'
            END AS user_segment
        FROM graded
    """)

    print("  分层结果：")
    show(con.execute("""
        SELECT user_segment AS 用户分层, COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
               ROUND(AVG(recency), 1)     AS 平均最近间隔天数,
               ROUND(AVG(active_days), 1) AS 平均活跃天数,
               ROUND(AVG(buy_pv), 1)      AS 平均购买次数
        FROM ads_user_segment GROUP BY 1 ORDER BY 用户数 DESC
    """).fetchdf(), max_rows=20)

    print("\n  合理性检查：各分层的数值范围是否真的分化了？")
    show(con.execute("""
        SELECT user_segment AS 用户分层,
               COUNT(*) AS 人数,
               MIN(recency) AS R最小, MAX(recency) AS R最大,
               MIN(active_days) AS F最小, MAX(active_days) AS F最大,
               MIN(buy_pv) AS M最小, MAX(buy_pv) AS M最大
        FROM ads_user_segment
        GROUP BY user_segment
        ORDER BY COUNT(*) DESC
    """).fetchdf(), max_rows=20)

    # ==========================================================
    # 【附加】各人群的运营策略建议
    # ==========================================================
    section("【附加】分层 → 运营策略映射")
    print("  输出：每类人群的规模、特征、建议动作\n")

    show(con.execute("""
        SELECT
            user_segment AS 人群,
            COUNT(*)     AS 规模,
            ROUND(AVG(active_days), 1) AS 平均活跃天数,
            ROUND(AVG(buy_pv), 1)      AS 平均购买次数,
            CASE user_segment
                WHEN '核心价值用户'   THEN 'VIP 维护：专属客服、优先发货、新品优先试用'
                WHEN '高频低转化用户' THEN '提升客单：推荐高价值商品、满减凑单'
                WHEN '低频高转化用户' THEN '提升频次：复购提醒、周期性商品推送'
                WHEN '新客'           THEN '首单转化：新手引导、首单立减'
                WHEN '一般活跃用户'   THEN '常规触达：内容营销、品类推荐'
                WHEN '流失预警用户'   THEN '召回优先：定向优惠券、唤醒短信'
                WHEN '已流失用户'     THEN '低成本触达：大促统一召回，不单独投放'
                ELSE '需求挖掘：分析浏览但未成交的原因'
            END AS 建议动作
        FROM ads_user_segment GROUP BY 1 ORDER BY 规模 DESC
    """).fetchdf(), max_rows=20)

    # ==========================================================
    # 【附加】商品质量分层的运营映射
    # ==========================================================
    section("【附加】商品质量分层 → 运营动作")
    print("  依据：商品级点击→加购率（要求点击 >= 200，保证样本量）\n")

    show(con.execute("""
        WITH q AS (
            SELECT * FROM ads_item_quality WHERE clicks >= 200
        )
        SELECT
            CASE
                WHEN cart_rate >= 8 THEN 'A-高意向商品'
                WHEN cart_rate >= 4 THEN 'B-中等意向商品'
                WHEN cart_rate >= 2 THEN 'C-低意向商品'
                ELSE 'D-需优化商品'
            END AS 商品分层,
            COUNT(*) AS 商品数,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
            SUM(clicks) AS 总点击,
            SUM(carts)  AS 总加购,
            SUM(buys)   AS 总支付,
            CASE
                WHEN cart_rate >= 8 THEN '加大推荐权重、保证库存'
                WHEN cart_rate >= 4 THEN '观察，优化详情页'
                WHEN cart_rate >= 2 THEN '重点优化：主图/标题/价格'
                ELSE '评估下架或重新选品'
            END AS 建议动作
        FROM q GROUP BY 1, 7 ORDER BY 1
    """).fetchdf(), max_rows=20)

    # ==========================================================
    # 导出
    # ==========================================================
    section("【导出】")
    import pandas as pd
    xlsx_path = config.TABLE_DIR / "P2fix_修正结果.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("SELECT * FROM ads_friction_cost").fetchdf().to_excel(
                writer, sheet_name="摩擦成本修正", index=False)
            con.execute("""
                SELECT user_segment AS 用户分层, COUNT(*) AS 用户数,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
                       ROUND(AVG(recency),1) AS 平均最近间隔天数,
                       ROUND(AVG(active_days),1) AS 平均活跃天数,
                       ROUND(AVG(buy_pv),1) AS 平均购买次数
                FROM ads_user_segment GROUP BY 1 ORDER BY 用户数 DESC
            """).fetchdf().to_excel(writer, sheet_name="用户分层", index=False)
        print(f"  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal

    section("完成")
    print(f"  耗时: {time.time() - t0:.1f} 秒")
    print(f"  日志: {log_path}")
    print(DIVIDER)


if __name__ == "__main__":
    # sys.stderr 已被指向 tee，因此未捕获的异常和 traceback
    # 会同时写入控制台和 P2fix_运行日志.txt
    main()
