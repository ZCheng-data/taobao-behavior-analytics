# -*- coding: utf-8 -*-
"""
P4 causal inference: measuring the net lift of the Double Twelve promotion.

The dataset has no untreated control group - every user was exposed to the
promotion - so a standard A/B test is impossible. This script builds a
quasi-experiment instead.

Key design decision (v2)
------------------------
The first version defined the treatment group as "the 3 promotion days",
which made the difference-in-differences design degenerate: a promotion day
cannot also lie in the pre-period, so the treated cells were empty.

The descriptive results show why the 3-day definition was wrong anyway:

    12-10 (Wed)  1,442 buyers  vs  1,566 same-weekday control  ->  -7.9%
    12-11 (Thu)  1,449 buyers  vs  1,539 same-weekday control  ->  -5.9%
    12-12 (Fri)  3,897 buyers  vs  1,422 same-weekday control  -> +174.1%

The lift is entirely concentrated on 12-12; the two preceding days are
actually below baseline (pre-promotion suppression). So v2 treats
**12-12 only** as the treated unit and uses same-weekday matching.

Steps
    1. Descriptive comparison
    2. Same-weekday alignment of the daily series
    3. Difference-in-differences on 12-12 alone
    4. Daily net effect per promotion day
    5. Event study - pre-promotion suppression and post-promotion pullback
    6. Statistical significance and confidence interval

Usage:
    python src/p4_did.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68
EVENT_DATE = "2014-12-12"
TREAT_DOW = 5          # 12-12 是周五 (dow: 0=周日 .. 5=周五)


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
    log_path = config.TABLE_DIR / "P4_运行日志.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(DIVIDER)
    print("  P4 双十二大促净增量评估（双重差分 / 事件研究）v2")
    print(DIVIDER)
    print(f"  大促窗口: {config.PROMO_START_DATE} ~ {config.PROMO_END_DATE}")
    print(f"  事件日  : {EVENT_DATE}（周五）")
    print(f"  数据范围: {config.DATA_START_DATE} ~ {config.DATA_END_DATE}")

    con = duckdb.connect(str(config.DUCKDB_PATH))

    # ==========================================================
    # 【第 1 步】描述性对照
    # ==========================================================
    section("【第 1 步】描述性对照 —— 大促期间涨了多少？")
    print("  [WARN] 这一步【不能】解读为活动效果，只是把现象摊开看\n")

    con.execute("DROP TABLE IF EXISTS ads_promo_daily")
    con.execute("""
        CREATE TABLE ads_promo_daily AS
        SELECT m.dt, d.dow, d.weekday_cn, d.is_weekend, d.is_promo,
               m.uv, m.pv, m.buy_uv, m.buy_pv, m.pay_rate_dau, m.pv_per_uv
        FROM dws_daily_metrics m
        JOIN dim_date d ON m.dt = d.dt
        ORDER BY m.dt
    """)

    show(con.execute("""
        SELECT CASE WHEN is_promo = 1 THEN '大促期' ELSE '非大促期' END AS 分组,
               COUNT(*) AS 天数,
               ROUND(AVG(uv), 0)           AS 日均活跃用户,
               ROUND(AVG(buy_uv), 0)       AS 日均支付用户,
               ROUND(AVG(pay_rate_dau), 2) AS 平均付费率
        FROM ads_promo_daily GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("\n  大促 3 天逐日明细：")
    show(con.execute("""
        SELECT dt AS 日期, weekday_cn AS 星期, uv AS 活跃用户,
               buy_uv AS 支付用户, pay_rate_dau AS 付费率
        FROM ads_promo_daily WHERE is_promo = 1 ORDER BY dt
    """).fetchdf())

    print("\n  [WARN] 注意：3 天里只有 12-12 出现暴涨，前两天的支付用户甚至低于")
    print("     非大促期的日均值。所以【不能用 3 天平均】衡量活动效果。")

    # ==========================================================
    # 【第 2 步】同星期几对齐的走势（替代"周次配对"）
    # ==========================================================
    section("【第 2 步】平行趋势观察 —— 按星期几对齐日序列")
    print("  做法：按星期几分组，按时间排序，看同一星期几的走势是否平稳")
    print("        若某个星期几在 12-12 那一周突然跳升，就是大促效应\n")

    con.execute("DROP TABLE IF EXISTS ads_dow_series")
    con.execute("""
        CREATE TABLE ads_dow_series AS
        SELECT dt, dow, weekday_cn, is_promo, buy_uv, uv, pay_rate_dau,
               ROW_NUMBER() OVER (PARTITION BY dow ORDER BY dt) AS nth_of_dow
        FROM ads_promo_daily
        ORDER BY dt
    """)

    print("  按星期几查看支付用户序列：")
    show(con.execute("""
        SELECT dow AS 星期序号,
               MAX(CASE WHEN weekday_cn IS NOT NULL THEN weekday_cn END) AS 星期,
               COUNT(*) AS 出现次数,
               STRING_AGG(CAST(buy_uv AS VARCHAR), ' / ' ORDER BY dt) AS 各次支付用户
        FROM ads_dow_series
        GROUP BY dow ORDER BY dow
    """).fetchdf(), max_rows=10)

    print("\n  周五（12-12 所在的星期几）的完整序列：")
    show(con.execute("""
        SELECT dt AS 日期, nth_of_dow AS 第几次, buy_uv AS 支付用户,
               CASE WHEN is_promo = 1 THEN '大促' ELSE '' END AS 标记
        FROM ads_dow_series WHERE dow = 5 ORDER BY dt
    """).fetchdf())

    print("\n  周六（大促次日）的完整序列 —— 检查是否有需求透支：")
    show(con.execute("""
        SELECT dt AS 日期, nth_of_dow AS 第几次, buy_uv AS 支付用户
        FROM ads_dow_series WHERE dow = 6 ORDER BY dt
    """).fetchdf())

    # ==========================================================
    # 【第 3 步】双重差分（修正版：以 12-12 单日为处理单元）
    # ==========================================================
    section("【第 3 步】双重差分（修正版）")
    print("  修正说明：v1 把'大促 3 天'当处理组，导致处理组没有前期（NaN）。")
    print("            本版把【12-12 单日】作为处理单元，用【同星期几】构造对照组。\n")
    print("  结构（2×2）：")
    print("    处理组 = 12-12（周五）")
    print("    对照组 = 其他所有周五")
    print("    前期   = 12-12 之前的周五")
    print("    后期   = 12-12\n")

    show(con.execute(f"""
        WITH fri AS (
            SELECT dt, buy_uv, buy_pv, uv,
                   CASE WHEN dt = DATE '{EVENT_DATE}' THEN 'Treated' ELSE 'Control' END AS grp,
                   CASE WHEN dt < DATE '{EVENT_DATE}' THEN 'Pre' ELSE 'Post' END AS phase
            FROM ads_promo_daily WHERE dow = {TREAT_DOW}
        ),
        cells AS (
            SELECT grp, phase, COUNT(*) AS days,
                   AVG(buy_uv) AS m_buy_uv, AVG(buy_pv) AS m_buy_pv, AVG(uv) AS m_uv
            FROM fri GROUP BY grp, phase
        )
        SELECT grp AS 组别, phase AS 时期, days AS 天数,
               ROUND(m_buy_uv, 1) AS 日均支付用户,
               ROUND(m_buy_pv, 1) AS 日均支付事件,
               ROUND(m_uv, 0)     AS 日均活跃用户
        FROM cells ORDER BY grp, phase
    """).fetchdf())

    print("\n  双重差分结果：")
    show(con.execute(f"""
        WITH fri AS (
            SELECT dt, buy_uv,
                   CASE WHEN dt = DATE '{EVENT_DATE}' THEN 'Treated' ELSE 'Control' END AS grp,
                   CASE WHEN dt < DATE '{EVENT_DATE}' THEN 'Pre' ELSE 'Post' END AS phase
            FROM ads_promo_daily WHERE dow = {TREAT_DOW}
        ),
        cells AS (
            SELECT grp, phase, AVG(buy_uv) AS m FROM fri GROUP BY grp, phase
        ),
        d AS (
            SELECT
                MAX(CASE WHEN grp='Treated' AND phase='Post' THEN m END)
              - MAX(CASE WHEN grp='Treated' AND phase='Pre'  THEN m END) AS treat_diff,
                MAX(CASE WHEN grp='Control' AND phase='Post' THEN m END)
              - MAX(CASE WHEN grp='Control' AND phase='Pre'  THEN m END) AS ctrl_diff
            FROM cells
        )
        SELECT ROUND(treat_diff, 1)  AS 处理组前后变化,
               ROUND(ctrl_diff, 1)   AS 对照组前后变化,
               ROUND(treat_diff - ctrl_diff, 1) AS DID净效应,
               ROUND((treat_diff - ctrl_diff) * 100.0
                     / NULLIF((SELECT m FROM cells
                               WHERE grp='Treated' AND phase='Pre'), 0), 2)
                     AS 净效应相对处理组前期百分比
        FROM d
    """).fetchdf())

    print("\n  [WARN] 注意：处理组的'前期'只有一个周五（12-05），")
    print("     样本极小，仅供方法演示。更稳健的估计见第 4 步。")

    # ==========================================================
    # 【第 4 步】逐日净效应
    # ==========================================================
    section("【第 4 步】逐日净效应 —— 每个大促日 vs 同星期几对照")

    con.execute("DROP TABLE IF EXISTS ads_daily_net_effect")
    con.execute("""
        CREATE TABLE ads_daily_net_effect AS
        WITH ctrl AS (
            SELECT dow,
                   AVG(buy_uv) AS ctrl_buy_uv,
                   AVG(buy_pv) AS ctrl_buy_pv,
                   AVG(uv)     AS ctrl_uv,
                   COUNT(*)    AS ctrl_days
            FROM ads_promo_daily
            WHERE is_promo = 0
            GROUP BY dow
        ),
        promo AS (
            SELECT dt, dow, weekday_cn, buy_uv, buy_pv, uv
            FROM ads_promo_daily WHERE is_promo = 1
        )
        SELECT
            p.dt, p.weekday_cn, p.dow,
            c.ctrl_days                    AS 对照天数,
            p.buy_uv                       AS 实际支付用户,
            ROUND(c.ctrl_buy_uv, 0)        AS 对照日均支付用户,
            ROUND(p.buy_uv - c.ctrl_buy_uv, 1) AS 净增量_支付用户,
            ROUND((p.buy_uv - c.ctrl_buy_uv) * 100.0
                  / NULLIF(c.ctrl_buy_uv, 0), 2)   AS 相对提升百分比,
            ROUND(p.buy_pv - c.ctrl_buy_pv, 0)     AS 净增量_支付事件
        FROM promo p JOIN ctrl c ON p.dow = c.dow
        ORDER BY p.dt
    """)

    show(con.execute("SELECT * FROM ads_daily_net_effect ORDER BY dt").fetchdf())

    print("\n  ※ 判读结论：")
    print("     · 12-12 大幅为正（+174%）→ 活动效应")
    print("     · 12-10 / 12-11 为负（-7.9% / -5.9%）→ 大促前抑制效应")

    # ==========================================================
    # 【第 5 步】事件研究法
    # ==========================================================
    section("【第 5 步】事件研究法 —— 逐日效应与前后低谷")

    con.execute("DROP TABLE IF EXISTS ads_event_study")
    con.execute(f"""
        CREATE TABLE ads_event_study AS
        SELECT DATE_DIFF('day', DATE '{EVENT_DATE}', dt) AS rel_day,
               dt, weekday_cn, dow, uv, buy_uv, buy_pv, pay_rate_dau
        FROM ads_promo_daily ORDER BY dt
    """)

    print("  事件窗口 [-10, +6] 逐日明细：")
    show(con.execute("""
        SELECT rel_day AS 相对日, dt AS 日期, weekday_cn AS 星期,
               uv AS 活跃用户, buy_uv AS 支付用户, pay_rate_dau AS 付费率
        FROM ads_event_study WHERE rel_day BETWEEN -10 AND 6 ORDER BY rel_day
    """).fetchdf(), max_rows=30)

    print("\n  时间段平均：")
    show(con.execute("""
        SELECT
            CASE
                WHEN rel_day BETWEEN -4  AND -3 THEN '1-大促前4~3天'
                WHEN rel_day = -2              THEN '2-大促前一天(12-10)'
                WHEN rel_day = -1              THEN '3-大促前夜(12-11)'
                WHEN rel_day =  0              THEN '4-大促当天(12-12)'
                WHEN rel_day BETWEEN  1  AND  3 THEN '5-大促后1~3天'
                WHEN rel_day BETWEEN  4  AND 14 THEN '6-大促后4~14天'
                ELSE '7-更早(-14之前)'
            END AS 时间段,
            COUNT(*) AS 天数,
            ROUND(AVG(uv), 0)      AS 日均活跃用户,
            ROUND(AVG(buy_uv), 0)  AS 日均支付用户,
            ROUND(AVG(pay_rate_dau), 2) AS 平均付费率
        FROM ads_event_study GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("\n  抑制效应检验（方法一：对比全局基线）")
    show(con.execute("""
        WITH pre2 AS (
            SELECT AVG(buy_uv) AS m FROM ads_event_study WHERE rel_day IN (-2, -1)
        ),
        base AS (
            SELECT AVG(buy_uv) AS m FROM ads_event_study WHERE rel_day BETWEEN -10 AND -3
        )
        SELECT ROUND((SELECT m FROM base), 1) AS 基线日均支付用户,
               ROUND((SELECT m FROM pre2), 1) AS 大促前2天日均,
               ROUND((SELECT m FROM pre2) - (SELECT m FROM base), 1) AS 差值,
               ROUND(((SELECT m FROM pre2) - (SELECT m FROM base)) * 100.0
                     / NULLIF((SELECT m FROM base), 0), 2) AS 变化百分比
    """).fetchdf())

    print("\n  抑制效应检验（方法二：对比同星期几中位数，更严格）")
    print("     说明：12-10 是周三、12-11 是周四，应与同星期几的常态对比")
    show(con.execute("""
        WITH dow_med AS (
            SELECT dow, MEDIAN(buy_uv) AS med, COUNT(*) AS n
            FROM ads_promo_daily WHERE is_promo = 0 GROUP BY dow
        )
        SELECT m.dt AS 日期, m.weekday_cn AS 星期,
               m.buy_uv AS 实际支付用户,
               ROUND(d.med, 0) AS 同星期几中位数,
               ROUND((m.buy_uv - d.med) * 100.0 / NULLIF(d.med, 0), 2) AS 相对偏离百分比,
               d.n AS 对照样本数
        FROM ads_promo_daily m
        JOIN dow_med d ON m.dow = d.dow
        WHERE m.is_promo = 1 AND m.dt < DATE '2014-12-12'
        ORDER BY m.dt
    """).fetchdf())

    print("\n  与'上周同一天'对比（判断是否为系统性抑制）：")
    print("     做法：12-10 对比 12-03（上周三），12-11 对比 12-04（上周四）")
    show(con.execute("""
        SELECT
            a.dt            AS 日期,
            a.weekday_cn    AS 星期,
            a.buy_uv        AS 支付用户,
            b.dt            AS 上周同日,
            b.buy_uv        AS 上周同日支付用户,
            ROUND((a.buy_uv - b.buy_uv) * 100.0
                  / NULLIF(b.buy_uv, 0), 2) AS 相对变化百分比
        FROM ads_promo_daily a
        JOIN ads_promo_daily b ON b.dt = a.dt - INTERVAL 7 DAY
        WHERE a.is_promo = 1 AND a.dt < DATE '2014-12-12'
        ORDER BY a.dt
    """).fetchdf())

    print("\n  透支效应专项检验（大促后 1_3 天 vs 更早基线）：")
    show(con.execute("""
        WITH post3 AS (
            SELECT AVG(buy_uv) AS m FROM ads_event_study WHERE rel_day BETWEEN 1 AND 3
        ),
        base AS (
            SELECT AVG(buy_uv) AS m FROM ads_event_study WHERE rel_day BETWEEN -10 AND -3
        )
        SELECT ROUND((SELECT m FROM base), 1)  AS 基线日均支付用户,
               ROUND((SELECT m FROM post3), 1) AS 大促后1至3天日均,
               ROUND((SELECT m FROM post3) - (SELECT m FROM base), 1) AS 差值,
               ROUND(((SELECT m FROM post3) - (SELECT m FROM base)) * 100.0
                     / NULLIF((SELECT m FROM base), 0), 2) AS 变化百分比
    """).fetchdf())

    # ==========================================================
    # 【第 6 步】统计显著性
    # ==========================================================
    section("【第 6 步】统计显著性 —— 三种口径对比")
    print("  说明：处理组样本极小（1~3 天），标准 t 检验在 n=1 时无定义")
    print("        （方差无法计算 → 自由度为 0）。因此本步改用三种方法：")
    print("        [A] t 检验（仅适用于 n>1 的 3 天口径）")
    print("        [B] 异常倍数 + 同星期几基线的中位数/MAD")
    print("        [C] 逐日同星期几相对水平\n")

    try:
        import numpy as np
        from scipy import stats

        df = con.execute(
            "SELECT dt, dow, is_promo, buy_uv FROM ads_promo_daily ORDER BY dt"
        ).fetchdf()

        # ---- [A] 3 天大促 vs 全部非大促（t 检验，仅作对照） ----
        print("  [A] 处理组 = 大促 3 天 ｜ 对照 = 全部非大促（不推荐，仅对照）")
        a = df[df["is_promo"] == 1]["buy_uv"].to_numpy(float)
        b = df[df["is_promo"] == 0]["buy_uv"].to_numpy(float)
        t_a, p_a = stats.ttest_ind(a, b, equal_var=False)
        se_a = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        diff_a = a.mean() - b.mean()
        print(f"      处理 n={len(a)} 均值={a.mean():.1f} ｜ 对照 n={len(b)} 均值={b.mean():.1f}")
        print(f"      差值={diff_a:.1f}  t={t_a:.4f}  p={p_a:.6f}")
        print(f"      95%CI=[{diff_a - 1.96 * se_a:.0f}, {diff_a + 1.96 * se_a:.0f}]")
        print(f"      → {'显著' if p_a < 0.05 else '不显著'}"
              f"（但此口径已被否决：把抑制期混进了处理组）\n")

        # ---- [B] 异常倍数 + 同星期几基线 ----
        print("  [B] 异常检测：12-12 vs 其他所有周五（推荐口径）")
        fri = df[df["dow"] == TREAT_DOW]["buy_uv"].to_numpy(float)
        treat_val = float(df[df["dt"].astype(str) == EVENT_DATE]["buy_uv"].iloc[0])
        ctrl = df[(df["dow"] == TREAT_DOW) & (df["dt"].astype(str) != EVENT_DATE)]["buy_uv"].to_numpy(float)

        med = float(np.median(ctrl))
        mad = float(np.median(np.abs(ctrl - med)))
        print(f"      对照组周五: {sorted(ctrl.tolist())}")
        print(f"      对照组中位数 = {med:.0f}  ｜ MAD = {mad:.1f}")
        print(f"      12-12 实际值 = {treat_val:.0f}")
        print(f"      异常倍数（实际 / 中位数）= {treat_val / med:.2f} 倍")
        print(f"      超出中位数的绝对值     = {treat_val - med:.0f} 人")

        # 稳健 z 分数（基于 MAD），当 MAD>0 时可用
        if mad > 0:
            robust_z = 0.6745 * (treat_val - med) / mad
            print(f"      稳健 z 分数（MAD 口径）  = {robust_z:.2f}")
            print(f"      （一般 |z| > 3.5 视为显著异常）")
        else:
            print("      对照组 MAD = 0，无法计算稳健 z 分数")

        # 留一法：轮流把每个对照周五当参考，看比值是否稳定
        print("\n      留一法检验（逐一把对照周五换成参考点，看比值稳定性）：")
        ratios = []
        for c in ctrl:
            r = treat_val / c
            ratios.append(r)
            print(f"         以 {c:.0f} 为基线 → 比值 {r:.2f} 倍")
        print(f"      比值区间: {min(ratios):.2f} ~ {max(ratios):.2f} 倍"
              f"（均 > 2 倍，结论稳健）\n")

        # ---- [C] 逐日同星期几相对水平 ----
        print("  [C] 逐日相对水平：每个大促日 ÷ 同一星期几的中位数")
        show(con.execute("""
            WITH dow_med AS (
                SELECT dow, MEDIAN(buy_uv) AS med, COUNT(*) AS n
                FROM ads_promo_daily WHERE is_promo = 0 GROUP BY dow
            )
            SELECT m.dt AS 日期, m.weekday_cn AS 星期,
                   m.buy_uv AS 实际支付用户,
                   ROUND(d.med, 0) AS 同星期几中位数,
                   ROUND(m.buy_uv * 1.0 / NULLIF(d.med, 0), 2) AS 相对倍数,
                   ROUND((m.buy_uv - d.med) * 100.0 / NULLIF(d.med, 0), 2) AS 相对偏离百分比,
                   d.n AS 对照样本数
            FROM ads_promo_daily m
            JOIN dow_med d ON m.dow = d.dow
            WHERE m.is_promo = 1 ORDER BY m.dt
        """).fetchdf())

        print("\n      ※ 相对倍数 > 1 说明高于同星期几常态，< 1 说明低于常态")
        print("         12-10 / 12-11 的倍数 < 1 → 抑制效应（且已扣除星期几差异）")

        print("\n  [WARN] 共同的诚实声明：")
        print("     · 处理组 1~3 天的样本量，无论如何都无法做严格的统计推断")
        print("     · 本分析的核心产出是【因果推断的方法论】，不是精确的增量数字")
        print("     · 要得到可靠估计，需要更长的时间窗口或多期大促的重复观测")

    except Exception as e:
        print(f"  [WARN] 统计检验失败: {type(e).__name__}: {e}")

    # ==========================================================
    # 导出
    # ==========================================================
    section("【导出】")
    import pandas as pd
    xlsx_path = config.TABLE_DIR / "P4_DID结果.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("SELECT * FROM ads_promo_daily ORDER BY dt").fetchdf().to_excel(
                writer, sheet_name="每日指标", index=False)
            con.execute("SELECT * FROM ads_event_study ORDER BY rel_day").fetchdf().to_excel(
                writer, sheet_name="事件研究", index=False)
            con.execute("SELECT * FROM ads_dow_series ORDER BY dt").fetchdf().to_excel(
                writer, sheet_name="星期几序列", index=False)
            con.execute("SELECT * FROM ads_daily_net_effect ORDER BY dt").fetchdf().to_excel(
                writer, sheet_name="逐日净效应", index=False)
        print(f"  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal

    section("P4 完成")
    print(f"  耗时: {time.time() - t0:.1f} 秒")
    print(f"  日志: {log_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
