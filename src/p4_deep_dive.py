# -*- coding: utf-8 -*-
"""
P4-deep: separating pre-promo pull-forward from pre-promo suppression using
the full daily series.
The problem
-----------
The observed change is a NET of two opposing forces:
    A. Pre-promo pull-forward -- users stock up early for 12-12, lifting the
       pre-promo baseline
    B. Pre-promo suppression   -- users know a discount is coming, delay the
       purchase, depressing the pre-promo baseline
Why the first two attempts failed
---------------------------------
- v1 used the whole-period mean as the baseline      -> "suppression of -5%"
- v2 used the same weekday of the previous week      -> "no suppression, +5%"
Both picked a single baseline, instead of recognising that the observed value
is the net sum of A and B.
What this script does
---------------------
1. Print the FULL daily series instead of only the last two points
2. Fit a linear trend over the 22 pre-promo days (11-18 ~ 12-09) to obtain a
   counterfactual baseline
3. Compute the residual = actual - counterfactual for every pre-promo day
4. Read the residuals by segment:
     11-18 ~ 11-27 (early)     residual should be near 0 (fitting window)
     11-28 ~ 12-05 (mid-late)  positive residual -> pull-forward
     12-06 ~ 12-09 (imminent)  negative residual -> suppression sets in
     12-10 ~ 12-11 (T-2, T-1)  strongly negative -> peak suppression
5. Extrapolate the same trend line to 12-12 for a cleaner counterfactual
Usage:
    python src/p4_deep_dive.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68
EVENT_DATE = "2014-12-12"


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


def show(df, max_rows=60):
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
    log_path = config.TABLE_DIR / "P4d_运行日志.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(DIVIDER)
    print("  P4 深入分析：分离「消费前移」与「大促前抑制」")
    print(DIVIDER)

    con = duckdb.connect(str(config.DUCKDB_PATH))

    # ==========================================================
    # 【第 1 步】完整日序列（不再只看最后两个点）
    # ==========================================================
    section("【第 1 步】完整日序列 —— 31 天全部输出")

    con.execute("DROP TABLE IF EXISTS p4d_series")
    con.execute("""
        CREATE TABLE p4d_series AS
        SELECT m.dt, d.dow, d.weekday_cn, d.is_promo,
               m.uv, m.buy_uv, m.buy_pv, m.pay_rate_dau,
               ROW_NUMBER() OVER (PARTITION BY d.dow ORDER BY m.dt) AS nth_of_dow
        FROM dws_daily_metrics m
        JOIN dim_date d ON m.dt = d.dt
        ORDER BY m.dt
    """)

    show(con.execute("""
        SELECT dt AS 日期, weekday_cn AS 星期, nth_of_dow AS 同星期第几次,
               uv AS 活跃用户, buy_uv AS 支付用户,
               CASE WHEN is_promo = 1 THEN '大促' ELSE '' END AS 标记
        FROM p4d_series ORDER BY dt
    """).fetchdf(), max_rows=40)

    # ==========================================================
    # 【第 2 步】按星期几看完整序列（判断趋势方向）
    # ==========================================================
    section("【第 2 步】按星期几看完整序列 —— 趋势到底是什么方向？")
    print("  [WARN] 这是上一轮我漏掉的关键：之前只看了每行的最后两个值\n")

    show(con.execute("""
        SELECT dow AS 星期序号,
               MAX(weekday_cn) AS 星期,
               COUNT(*) AS 出现次数,
               STRING_AGG(CAST(buy_uv AS VARCHAR), ' → ' ORDER BY dt) AS 支付用户完整序列,
               MAX(CASE WHEN is_promo = 1 THEN '含大促日' ELSE '纯对照' END) AS 序列性质
        FROM p4d_series
        GROUP BY dow ORDER BY dow
    """).fetchdf(), max_rows=10)

    print("\n  用【大促前】的完整序列计算同星期几的逐周变化率：")
    show(con.execute("""
        WITH pre AS (
            SELECT dow, weekday_cn, dt, nth_of_dow, buy_uv
            FROM p4d_series WHERE is_promo = 0
        )
        SELECT p1.dow AS 星期序号, p1.weekday_cn AS 星期,
               p1.buy_uv AS 第1次, p2.buy_uv AS 第2次, p3.buy_uv AS 第3次,
               ROUND((p2.buy_uv - p1.buy_uv) * 100.0 / NULLIF(p1.buy_uv, 0), 2) AS 第1到2次百分比,
               ROUND((p3.buy_uv - p2.buy_uv) * 100.0 / NULLIF(p2.buy_uv, 0), 2) AS 第2到3次百分比
        FROM pre p1
        JOIN pre p2 ON p2.dow = p1.dow AND p2.nth_of_dow = 2
        JOIN pre p3 ON p3.dow = p1.dow AND p3.nth_of_dow = 3
        WHERE p1.nth_of_dow = 1
        ORDER BY p1.dow
    """).fetchdf())

    # ==========================================================
    # 【第 3 步】同星期几中位数基线 + 窗口敏感性
    # ==========================================================
    section("【第 3 步】稳健基线 —— 同星期几中位数 + 窗口敏感性检验")
    print("  [WARN] 线性趋势拟合在 31 天数据上过于脆弱（R² 仅 0.16）")
    print("     与 12-03 的高点相抵触；改用【同星期几中位数】作基线。")
    print("     划分：前段 11-18~11-27 ｜ 基准窗口 11-18~12-05（扣除大促日）")
    print("           观测期 11-28~12-11\n")

    def period_label(col="dt"):
        return f"""
            CASE
                WHEN {col} BETWEEN DATE '2014-11-18' AND DATE '2014-11-27' THEN '1-前段校准(11-18~27)'
                WHEN {col} BETWEEN DATE '2014-11-28' AND DATE '2014-12-01' THEN '2-中后段(11-28~12-01)'
                WHEN {col} BETWEEN DATE '2014-12-02' AND DATE '2014-12-05' THEN '3-第3周初(12-02~05)'
                WHEN {col} BETWEEN DATE '2014-12-06' AND DATE '2014-12-09' THEN '4-大促前4天(12-06~09)'
                WHEN {col} = DATE '2014-12-10' THEN '5-大促前一天(12-10)'
                WHEN {col} = DATE '2014-12-11' THEN '6-大促前夜(12-11)'
                WHEN {col} = DATE '2014-12-12' THEN '7-大促当天(12-12)'
                WHEN {col} BETWEEN DATE '2014-12-13' AND DATE '2014-12-18' THEN '8-大促后(12-13~18)'
                ELSE '9-其他'
            END
        """

    print("  主基线：用 11-18 ~ 12-05 的同星期几中位数作为基线")
    show(con.execute(f"""
        WITH base AS (
            SELECT dow, MEDIAN(buy_uv) AS base_val, COUNT(*) AS n
            FROM p4d_series
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
            GROUP BY dow
        ),
        calc AS (
            SELECT s.dt, s.dow, s.weekday_cn, s.is_promo, s.buy_uv,
                   b.base_val, b.n,
                   s.buy_uv - b.base_val AS residual,
                   ROUND((s.buy_uv - b.base_val) * 100.0
                         / NULLIF(b.base_val, 0), 2) AS residual_pct,
                   {period_label("s.dt")} AS 时段
            FROM p4d_series s JOIN base b ON s.dow = b.dow
        )
        SELECT 时段, COUNT(*) AS 天数,
               ROUND(AVG(base_val), 0) AS 基线均值,
               ROUND(AVG(buy_uv), 0)   AS 实际均值,
               ROUND(AVG(residual), 0) AS 平均残差,
               ROUND(AVG(residual_pct), 2) AS 平均残差百分比
        FROM calc
        WHERE 时段 <> '9-其他'
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("\n  检查是否所有星期几都出现同一个形状（一致性检验）：")
    print("     做法：看每个星期几在【中后段/第3周初/大促前】的残差符号是否一致")
    con.execute("DROP TABLE IF EXISTS p4d_weekday_resid")
    con.execute(f"""
        CREATE TABLE p4d_weekday_resid AS
        WITH base AS (
            SELECT dow, MEDIAN(buy_uv) AS base_val
            FROM p4d_series
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
            GROUP BY dow
        )
        SELECT s.dow, s.weekday_cn, s.dt, s.buy_uv, b.base_val,
               ROUND((s.buy_uv - b.base_val) * 100.0
                     / NULLIF(b.base_val, 0), 1) AS pct,
               CASE
                   WHEN s.dt BETWEEN DATE '2014-11-28' AND DATE '2014-12-01' THEN '中后段'
                   WHEN s.dt BETWEEN DATE '2014-12-02' AND DATE '2014-12-05' THEN '第3周初'
                   WHEN s.dt BETWEEN DATE '2014-12-06' AND DATE '2014-12-09' THEN '大促前4天'
                   ELSE '大促前2天'
               END AS 时段
        FROM p4d_series s JOIN base b ON s.dow = b.dow
        WHERE s.dt BETWEEN DATE '2014-11-28' AND DATE '2014-12-11'
    """)

    show(con.execute("""
        SELECT 星期,
               MAX(CASE WHEN 时段 = '中后段'   THEN pct END) AS 中后段,
               MAX(CASE WHEN 时段 = '第3周初'  THEN pct END) AS 第3周初,
               MAX(CASE WHEN 时段 = '大促前4天' THEN pct END) AS 大促前4天,
               MAX(CASE WHEN 时段 = '大促前2天' THEN pct END) AS 大促前2天
        FROM (
            SELECT weekday_cn AS 星期, pct, 时段 FROM p4d_weekday_resid
        )
        GROUP BY 星期 ORDER BY 星期
    """).fetchdf(), max_rows=10)

    print("\n  窗口敏感性检验：换不同基线窗口，看结论是否稳定")
    print("     （若结论只在某个窗口下成立，说明是挑选出来的）")

    for label, start, end in [
        ("窗口1: 11-18~11-27（仅前段）", "2014-11-18", "2014-11-27"),
        ("窗口2: 11-25~12-05（中段）",   "2014-11-25", "2014-12-05"),
        ("窗口3: 11-18~12-05（全前段）", "2014-11-18", "2014-12-05"),
    ]:
        print(f"\n  --- {label} ---")
        show(con.execute(f"""
            WITH base AS (
                SELECT dow, MEDIAN(buy_uv) AS base_val
                FROM p4d_series
                WHERE dt BETWEEN DATE '{start}' AND DATE '{end}'
                GROUP BY dow
            )
            SELECT
                ROUND(AVG(CASE WHEN s.dt BETWEEN DATE '2014-12-02'
                               AND DATE '2014-12-05' THEN
                          (s.buy_uv - b.base_val) * 100.0 / b.base_val END), 2)
                    AS 第3周初残差百分比,
                ROUND(AVG(CASE WHEN s.dt BETWEEN DATE '2014-12-06'
                               AND DATE '2014-12-09' THEN
                          (s.buy_uv - b.base_val) * 100.0 / b.base_val END), 2)
                    AS 大促前4天残差百分比,
                ROUND(AVG(CASE WHEN s.dt BETWEEN DATE '2014-12-10'
                               AND DATE '2014-12-11' THEN
                          (s.buy_uv - b.base_val) * 100.0 / b.base_val END), 2)
                    AS 大促前2天残差百分比,
                ROUND(AVG(CASE WHEN s.dt = DATE '2014-12-12' THEN
                          (s.buy_uv - b.base_val) * 100.0 / b.base_val END), 2)
                    AS 大促当天残差百分比
            FROM p4d_series s JOIN base b ON s.dow = b.dow
        """).fetchdf())

    print("\n  线性趋势拟合（仅作对照，R² 低说明不可靠）：")
    show(con.execute(f"""
        WITH idx AS (
            SELECT dt, buy_uv,
                   DATE_DIFF('day', DATE '{config.DATA_START_DATE}', dt) AS t
            FROM p4d_series WHERE is_promo = 0
        )
        SELECT ROUND(REGR_SLOPE(buy_uv, t), 3)  AS 每日斜率,
               ROUND(REGR_SLOPE(buy_uv, t) * 7, 1) AS 折算周增长,
               ROUND(REGR_R2(buy_uv, t), 4)    AS 拟合优度R2,
               ROUND(CORR(buy_uv, t), 4)       AS 相关系数
        FROM idx
    """).fetchdf())
    print("  [WARN] R² 越低，说明线性趋势越不成立，越应依赖同星期几基线")

    # ==========================================================
    # 【第 4 步】用同星期几基线估计 12-12 的净增量
    # ==========================================================
    section("【第 4 步】用同星期几基线估计 12-12 净增量")

    print("  基线口径：11-18 ~ 12-05 期间该星期几的中位数\n")

    print("  四种反事实口径对比：")
    show(con.execute(f"""
        WITH base_med AS (
            SELECT dow, MEDIAN(buy_uv) AS med
            FROM p4d_series
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
            GROUP BY dow
        ),
        base_all AS (
            SELECT MEDIAN(buy_uv) AS med FROM p4d_series
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
        ),
        last_fri AS (
            SELECT buy_uv AS v FROM p4d_series WHERE dt = DATE '2014-12-05'
        ),
        act AS (
            SELECT buy_uv AS a, dow FROM p4d_series WHERE dt = DATE '{EVENT_DATE}'
        )
        SELECT
            (SELECT a FROM act)                          AS 当日实际,
            ROUND((SELECT med FROM base_med b
                   WHERE b.dow = (SELECT dow FROM act)), 0) AS 口径A_同星期几中位数,
            ROUND((SELECT a FROM act)
                  - (SELECT med FROM base_med b
                     WHERE b.dow = (SELECT dow FROM act)), 0) AS 口径A_净增量,
            ROUND((SELECT v FROM last_fri), 0)           AS 口径B_上周五,
            ROUND((SELECT a FROM act)
                  - (SELECT v FROM last_fri), 0)         AS 口径B_净增量,
            ROUND((SELECT med FROM base_all), 0)         AS 口径C_全期前段中位数,
            ROUND((SELECT a FROM act)
                  - (SELECT med FROM base_all), 0)       AS 口径C_净增量
    """).fetchdf())

    print("\n  [WARN] 三种口径的差异来自【对反事实的不同假设】：")
    print("     口径 A（同星期几中位数）扣除了星期几效应，也扣除了部分趋势 → 最推荐")
    print("     口径 B（上周五）最贴近事件时点，但受单日波动影响大")
    print("     口径 C（全期前段中位数）未区分星期几 → 会系统性偏差")

    # ==========================================================
    # 【第 5 步】分段累计效应
    # ==========================================================
    section("【第 5 步】分段累计效应 —— 两股力量各有多大？")
    print("  分段口径与第 3 步保持一致：")
    print("    消费前移期 12-02~12-05 ｜ 抑制期 12-06~12-11")
    print("    大促当天 12-12        ｜ 大促后 12-13~12-18\n")

    show(con.execute("""
        WITH base AS (
            SELECT dow, MEDIAN(buy_uv) AS base_val
            FROM p4d_series
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
            GROUP BY dow
        ),
        calc AS (
            SELECT s.dt, s.buy_uv, b.base_val,
                   s.buy_uv - b.base_val AS resid
            FROM p4d_series s JOIN base b ON s.dow = b.dow
        )
        SELECT 'A-消费前移期(12-02~12-05)' AS 时期, COUNT(*) AS 天数,
               ROUND(SUM(buy_uv), 0)   AS 实际累计,
               ROUND(SUM(base_val), 0) AS 基线累计,
               ROUND(SUM(resid), 0)    AS 累计差额
        FROM calc WHERE dt BETWEEN DATE '2014-12-02' AND DATE '2014-12-05'
        UNION ALL
        SELECT 'B-抑制期(12-06~12-11)', COUNT(*),
               ROUND(SUM(buy_uv), 0), ROUND(SUM(base_val), 0), ROUND(SUM(resid), 0)
        FROM calc WHERE dt BETWEEN DATE '2014-12-06' AND DATE '2014-12-11'
        UNION ALL
        SELECT 'C-大促当天(12-12)', COUNT(*),
               ROUND(SUM(buy_uv), 0), ROUND(SUM(base_val), 0), ROUND(SUM(resid), 0)
        FROM calc WHERE dt = DATE '2014-12-12'
        UNION ALL
        SELECT 'D-大促后(12-13~12-18)', COUNT(*),
               ROUND(SUM(buy_uv), 0), ROUND(SUM(base_val), 0), ROUND(SUM(resid), 0)
        FROM calc WHERE dt BETWEEN DATE '2014-12-13' AND DATE '2014-12-18'
    """).fetchdf())

    print("\n  全期间净效应汇总（各段残差直接相加）：")
    show(con.execute("""
        WITH base AS (
            SELECT dow, MEDIAN(buy_uv) AS base_val
            FROM p4d_series
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
            GROUP BY dow
        ),
        calc AS (
            SELECT s.dt, s.buy_uv, b.base_val, s.buy_uv - b.base_val AS resid
            FROM p4d_series s JOIN base b ON s.dow = b.dow
        )
        SELECT
            ROUND(SUM(CASE WHEN dt BETWEEN DATE '2014-12-02'
                           AND DATE '2014-12-05' THEN resid END), 0) AS 前移期净增,
            ROUND(SUM(CASE WHEN dt BETWEEN DATE '2014-12-06'
                           AND DATE '2014-12-11' THEN resid END), 0) AS 抑制期净减,
            ROUND(SUM(CASE WHEN dt = DATE '2014-12-12' THEN resid END), 0) AS 当日净增,
            ROUND(SUM(CASE WHEN dt BETWEEN DATE '2014-12-13'
                           AND DATE '2014-12-18' THEN resid END), 0) AS 大促后净增,
            ROUND(SUM(CASE WHEN dt BETWEEN DATE '2014-12-02'
                           AND DATE '2014-12-18' THEN resid END), 0) AS 全窗口净效应
        FROM calc
    """).fetchdf())

    print("\n  解释：净增量的来源拆解")
    print("     预期看到：前移期净增 > 0，抑制期净减 < 0，当日净增 ≫ 0")
    print("     若【全窗口净效应】≫ 0 → 增量是真实的，不只是跨期替代")

    # ==========================================================
    # 导出
    # ==========================================================
    section("【导出】")
    import pandas as pd
    xlsx_path = config.TABLE_DIR / "P4d_深入分析.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("SELECT * FROM p4d_series ORDER BY dt").fetchdf().to_excel(
                writer, sheet_name="完整日序列", index=False)
            con.execute("SELECT * FROM p4d_weekday_resid ORDER BY dt").fetchdf().to_excel(
                writer, sheet_name="星期几残差", index=False)
            con.execute("""
                WITH base AS (
                    SELECT dow, MEDIAN(buy_uv) AS base_val
                    FROM p4d_series
                    WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-12-05'
                    GROUP BY dow
                )
                SELECT s.dt AS 日期, s.weekday_cn AS 星期, s.buy_uv AS 实际,
                       ROUND(b.base_val, 0) AS 基线,
                       ROUND(s.buy_uv - b.base_val, 0) AS 残差,
                       ROUND((s.buy_uv - b.base_val) * 100.0 / b.base_val, 2) AS 残差百分比
                FROM p4d_series s JOIN base b ON s.dow = b.dow
                ORDER BY s.dt
            """).fetchdf().to_excel(writer, sheet_name="逐日残差", index=False)
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
    main()
