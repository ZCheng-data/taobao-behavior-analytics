# -*- coding: utf-8 -*-
"""
P5-feasibility: is a churn prediction model meaningful on this dataset?

Before building any model, this script checks whether the data can support
a churn prediction task at all. Four diagnostics:

  1. How much of the "churn" is just the end-of-window truncation effect?
     If most users' last activity is near 2014-12-18, they did not churn -
     the observation window simply ended.

  2. How many users show genuine activity decay (active early, silent later)?

  3. What is the churn rate under different definitions, and is the class
     balance usable for modelling?

  4. If we predict D8-D14 from D1-D7, what is the naive baseline accuracy?
     A model is only worth building if it beats that baseline.

Usage:
    python src/p5_feasibility.py
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
    log_path = config.TABLE_DIR / "P5可行性诊断_运行日志.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(DIVIDER)
    print("  P5 可行性诊断：这个数据集适合做流失预警吗？")
    print(DIVIDER)
    print(f"  数据范围: {config.DATA_START_DATE} ~ {config.DATA_END_DATE}（31 天）")
    print(f"  用户规模: 10,000 人")

    con = duckdb.connect(str(config.DUCKDB_PATH))

    # ==========================================================
    # 【第 0 步】前置检查：确认依赖表的列名
    # ==========================================================
    section("【第 0 步】前置检查 —— 依赖表的列名")
    print("  目的：避免跑到一半才发现列名不匹配\n")

    for tbl in ("dws_user_daily", "dws_user_profile"):
        try:
            cols = con.execute(f"DESCRIBE {tbl}").fetchdf()
            names = ", ".join(cols["column_name"].tolist())
            print(f"  {tbl}:")
            print(f"    {names}\n")
        except Exception as e:
            print(f"  [FAIL] 表 {tbl} 不存在或不可读: {type(e).__name__}")
            print(f"     请先运行 p2_metrics.py 生成该表\n")
            con.close()
            sys.stdout = sys.stdout.terminal
            sys.stderr = sys.stderr.terminal
            return

    # ==========================================================
    # 【诊断 1】最后活跃日的分布 —— 是不是"窗口末尾效应"？
    # ==========================================================
    section("【诊断 1】用户最后活跃日分布 —— 区分真流失 vs 窗口截断")

    con.execute("DROP TABLE IF EXISTS p5_last_active")
    con.execute(f"""
        CREATE TABLE p5_last_active AS
        SELECT
            user_id,
            first_dt,
            last_dt,
            active_days,
            total_pv,
            total_buy_pv,
            DATE_DIFF('day', last_dt, DATE '{config.DATA_END_DATE}') AS days_since_last
        FROM dws_user_profile
    """)

    print("  表结构核对（dws_user_profile 的实际列）：")
    show(con.execute("DESCRIBE p5_last_active").fetchdf())

    print("  距最后活跃日的天数分布（days_since_last）：")
    show(con.execute("""
        SELECT
            CASE
                WHEN days_since_last = 0  THEN '0-最后一天仍活跃'
                WHEN days_since_last <= 3 THEN '1-3天没来'
                WHEN days_since_last <= 7 THEN '4-7天没来'
                WHEN days_since_last <= 14 THEN '8-14天没来'
                ELSE '15天以上没来'
            END AS 区间,
            COUNT(*) AS 用户数,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
            ROUND(AVG(active_days), 1) AS 平均活跃天数
        FROM p5_last_active
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("\n  ※ 关键判读：")
    print("     若'最后一天仍活跃'占比很高 → 大量用户的活跃是被【窗口截断】的")
    print("     这类用户不是流失，而是观测期结束了")

    print("\n  逐日看最后活跃日的分布（是否集中在末尾）：")
    show(con.execute(f"""
        SELECT last_dt AS 最后活跃日,
               COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM p5_last_active), 2) AS 占比百分比,
               CASE WHEN last_dt >= DATE '2014-12-16' THEN '← 窗口末尾' ELSE '' END AS 标记
        FROM p5_last_active
        GROUP BY last_dt ORDER BY last_dt DESC
    """).fetchdf(), max_rows=40)

    # ==========================================================
    # 【诊断 2】真实的活跃度衰减 —— 前后半段对比
    # ==========================================================
    section("【诊断 2】真实的活跃度衰减 —— 前后半段对比")
    print("  做法：把 31 天切成前 15 天 / 后 16 天，比较每人的活跃天数\n")

    con.execute("DROP TABLE IF EXISTS p5_decay")
    con.execute(f"""
        CREATE TABLE p5_decay AS
        WITH half AS (
            SELECT user_id,
                   SUM(CASE WHEN dt <= DATE '2014-12-02' THEN 1 ELSE 0 END) AS 前半段活跃天数,
                   SUM(CASE WHEN dt >  DATE '2014-12-02' THEN 1 ELSE 0 END) AS 后半段活跃天数,
                   SUM(CASE WHEN dt <= DATE '2014-12-02' THEN pv ELSE 0 END)  AS 前半段PV,
                   SUM(CASE WHEN dt >  DATE '2014-12-02' THEN pv ELSE 0 END)  AS 后半段PV
            FROM dws_user_daily GROUP BY user_id
        )
        SELECT user_id, 前半段活跃天数, 后半段活跃天数, 前半段PV, 后半段PV,
               后半段活跃天数 - 前半段活跃天数 AS 活跃天数变化,
               CASE
                   WHEN 前半段活跃天数 = 0 THEN 'X-前半段未出现'
                   WHEN 后半段活跃天数 = 0 THEN 'A-完全停止(真流失)'
                   WHEN 后半段活跃天数 * 1.0 / NULLIF(前半段活跃天数, 0) < 0.5
                        THEN 'B-大幅衰减(>50%降幅)'
                   WHEN 后半段活跃天数 * 1.0 / NULLIF(前半段活跃天数, 0) < 0.8
                        THEN 'C-中度衰减'
                   ELSE 'D-稳定或增长'
               END AS 衰减类型
        FROM half
    """)

    print("  衰减类型分布：")
    show(con.execute("""
        SELECT 衰减类型 AS 类型, COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比,
               ROUND(AVG(前半段活跃天数), 1) AS 前半段平均活跃,
               ROUND(AVG(后半段活跃天数), 1) AS 后半段平均活跃
        FROM p5_decay GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("\n  ※ 关键判读：")
    print("     若'A-完全停止'和'B-大幅衰减'合计占比 < 5%")
    print("     → 可预警的对象太少，模型没有实用价值")

    # ==========================================================
    # 【诊断 3】不同流失定义下的流失率
    # ==========================================================
    section("【诊断 3】不同流失定义下的流失率（判断正负样本平衡）")
    print("  注意：天数越大，被计入'流失'的用户越多，但其中包含越多的窗口截断效应\n")

    print("\n  不同阈值下的'末尾 N 天未活跃'人数（累积口径）：")
    show(con.execute("""
        WITH th AS (
            SELECT * FROM (VALUES (1), (3), (7), (14)) t(n)
        )
        SELECT t.n AS 阈值_天,
               SUM(CASE WHEN p.days_since_last >= t.n THEN 1 ELSE 0 END) AS 用户数,
               ROUND(SUM(CASE WHEN p.days_since_last >= t.n THEN 1 ELSE 0 END)
                     * 100.0 / (SELECT COUNT(*) FROM p5_last_active), 2) AS 占比百分比
        FROM th t CROSS JOIN p5_last_active p
        GROUP BY t.n ORDER BY t.n
    """).fetchdf())

    print("\n  按'连续 N 天无行为'定义的流失（含窗口内的空档 + 末尾空档）：")
    show(con.execute(f"""
        WITH u AS (
            SELECT user_id, dt FROM dws_user_daily
        ),
        spans AS (
            SELECT user_id, dt,
                   DATE_DIFF('day', dt,
                       LEAD(dt) OVER (PARTITION BY user_id ORDER BY dt)) AS gap
            FROM u
        ),
        max_gap AS (
            SELECT user_id,
                   COALESCE(MAX(gap), 1) AS max_gap_days
            FROM spans GROUP BY user_id
        ),
        tail_gap AS (
            SELECT user_id,
                   DATE_DIFF('day', last_dt, DATE '{config.DATA_END_DATE}') AS tail
            FROM p5_last_active
        )
        SELECT
            COUNT(*) AS 用户总数,
            SUM(CASE WHEN m.max_gap_days >= 7  THEN 1 ELSE 0 END) AS 窗口内有7天空档,
            SUM(CASE WHEN m.max_gap_days >= 14 THEN 1 ELSE 0 END) AS 窗口内有14天空档,
            SUM(CASE WHEN t.tail >= 7  THEN 1 ELSE 0 END) AS 末尾7天未活跃,
            SUM(CASE WHEN t.tail >= 14 THEN 1 ELSE 0 END) AS 末尾14天未活跃
        FROM max_gap m JOIN tail_gap t ON m.user_id = t.user_id
    """).fetchdf())

    # ==========================================================
    # 【诊断 4】D1-D7 预测 D8-D14 的可行性
    # ==========================================================
    section("【诊断 4】滑动窗口预测的可行性（D1-D7 → D8-D14）")
    print("  设计：")
    print("    特征窗口 = 11-18 ~ 11-24（7 天）")
    print("    标签窗口 = 11-25 ~ 12-01（7 天）")
    print("    标签定义 = 在标签窗口内【完全没有行为】\n")

    con.execute("DROP TABLE IF EXISTS p5_windows")
    con.execute("""
        CREATE TABLE p5_windows AS
        WITH feat AS (
            SELECT user_id,
                   COUNT(DISTINCT dt)  AS f_active_days,
                   SUM(pv)             AS f_pv,
                   SUM(click_pv)       AS f_click,
                   SUM(cart_pv)        AS f_cart,
                   SUM(fav_pv)         AS f_fav,
                   SUM(buy_pv)         AS f_buy,
                   SUM(active_hours)   AS f_hours,
                   MAX(item_cnt)       AS f_max_daily_items,
                   MIN(dt)             AS f_first,
                   MAX(dt)             AS f_last
            FROM dws_user_daily
            WHERE dt BETWEEN DATE '2014-11-18' AND DATE '2014-11-24'
            GROUP BY user_id
        ),
        label AS (
            SELECT user_id,
                   COUNT(DISTINCT dt) AS l_active_days,
                   SUM(buy_pv)        AS l_buy
            FROM dws_user_daily
            WHERE dt BETWEEN DATE '2014-11-25' AND DATE '2014-12-01'
            GROUP BY user_id
        )
        SELECT f.*,
               COALESCE(l.l_active_days, 0) AS l_active_days,
               COALESCE(l.l_buy, 0)         AS l_buy,
               CASE WHEN COALESCE(l.l_active_days, 0) = 0 THEN 1 ELSE 0 END AS label_churn
        FROM feat f LEFT JOIN label l ON f.user_id = l.user_id
    """)

    print("  样本与标签分布：")
    show(con.execute("""
        SELECT
            COUNT(*) AS 样本数,
            SUM(label_churn) AS 流失样本数,
            ROUND(SUM(label_churn) * 100.0 / COUNT(*), 2) AS 流失率百分比,
            COUNT(*) - SUM(label_churn) AS 未流失样本数,
            ROUND((COUNT(*) - SUM(label_churn)) * 100.0 / COUNT(*), 2) AS 未流失占比
        FROM p5_windows
    """).fetchdf())

    print("\n  关键：基准准确率（naive baseline）")
    print("     若模型只会说'全部不流失'，它的准确率是多少？")
    show(con.execute("""
        WITH s AS (SELECT COUNT(*) AS n, SUM(label_churn) AS c FROM p5_windows)
        SELECT n AS 样本数, c AS 流失数,
               ROUND(100.0 - c * 100.0 / n, 2) AS 全预测不流失的准确率,
               ROUND(c * 100.0 / n, 2)         AS 全预测流失的准确率
        FROM s
    """).fetchdf())

    print("\n  ※ 判读：")
    print("     若'全预测不流失'的准确率 > 90% → 模型必须显著超过这个数才有价值")
    print("     且流失率若 < 5% 或 > 95% → 正负样本极端不平衡，建模意义有限")

    print("\n  特征窗口内的活跃度 vs 标签（看是否真有区分度）：")
    show(con.execute("""
        SELECT label_churn AS 是否流失,
               COUNT(*) AS 用户数,
               ROUND(AVG(f_active_days), 2) AS 特征窗口活跃天数,
               ROUND(AVG(f_pv), 0)          AS 特征窗口PV,
               ROUND(AVG(f_cart), 2)        AS 特征窗口加购,
               ROUND(AVG(f_buy), 2)         AS 特征窗口购买
        FROM p5_windows GROUP BY label_churn ORDER BY label_churn
    """).fetchdf())

    print("\n  按特征窗口活跃天数分档，看流失率如何变化：")
    show(con.execute("""
        SELECT f_active_days AS 特征窗口活跃天数,
               COUNT(*) AS 用户数,
               SUM(label_churn) AS 流失数,
               ROUND(SUM(label_churn) * 100.0 / COUNT(*), 2) AS 流失率百分比
        FROM p5_windows GROUP BY f_active_days ORDER BY f_active_days
    """).fetchdf(), max_rows=15)

    # ==========================================================
    # 【诊断 5】换个窗口再试一次（稳健性）
    # ==========================================================
    section("【诊断 5】换个窗口验证（11-25~12-01 → 12-02~12-08）")

    show(con.execute("""
        WITH feat AS (
            SELECT user_id, COUNT(DISTINCT dt) AS f_active_days, SUM(pv) AS f_pv
            FROM dws_user_daily WHERE dt BETWEEN DATE '2014-11-25' AND DATE '2014-12-01'
            GROUP BY user_id
        ),
        label AS (
            SELECT user_id, COUNT(DISTINCT dt) AS l_active
            FROM dws_user_daily WHERE dt BETWEEN DATE '2014-12-02' AND DATE '2014-12-08'
            GROUP BY user_id
        )
        SELECT COUNT(*) AS 样本数,
               SUM(CASE WHEN COALESCE(l.l_active,0) = 0 THEN 1 ELSE 0 END) AS 流失样本数,
               ROUND(SUM(CASE WHEN COALESCE(l.l_active,0) = 0 THEN 1 ELSE 0 END)
                     * 100.0 / COUNT(*), 2) AS 流失率百分比
        FROM feat f LEFT JOIN label l ON f.user_id = l.user_id
    """).fetchdf())

    # ==========================================================
    # 【结论】可行性判定
    # ==========================================================
    section("【结论】可行性判定")

    stats = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM p5_last_active) AS total_users,
            (SELECT COUNT(*) FROM p5_last_active
             WHERE days_since_last = 0) AS still_active,
            (SELECT COUNT(*) FROM p5_decay
             WHERE 衰减类型 IN ('A-完全停止(真流失)', 'B-大幅衰减(>50%降幅)')) AS real_decay,
            (SELECT COUNT(*) FROM p5_windows) AS win_n,
            (SELECT SUM(label_churn) FROM p5_windows) AS win_churn
    """).fetchdf()

    total = int(stats.loc[0, "total_users"])
    still = int(stats.loc[0, "still_active"])
    decay = int(stats.loc[0, "real_decay"])
    win_n = int(stats.loc[0, "win_n"])
    win_c = int(stats.loc[0, "win_churn"])

    print(f"""
  ① 窗口截断效应
     最后一天（12-18）仍活跃的用户: {still:,} / {total:,} = {still*100.0/total:.1f}%

  ② 真实活跃度衰减
     完全停止 + 大幅衰减的用户: {decay:,} / {total:,} = {decay*100.0/total:.1f}%

  ③ 滑动窗口预测的正负样本
     样本数 = {win_n:,}，流失样本 = {win_c:,}，流失率 = {win_c*100.0/win_n:.2f}%
     全预测"不流失"的基线准确率 = {100.0 - win_c*100.0/win_n:.2f}%
    """)

    print(DIVIDER)
    print("  判定")
    print(DIVIDER)

    verdict = []
    if still * 100.0 / total > 60:
        verdict.append("[FAIL] 窗口截断效应严重：超过 60% 的用户在最后一天仍活跃，")
        verdict.append("   '流失'很大程度是观测期结束造成的假象")
    if decay * 100.0 / total < 5:
        verdict.append("[FAIL] 真实衰减用户过少：可预警对象不足 5%")
    if win_c * 100.0 / win_n < 5:
        verdict.append("[WARN] 标签极端不平衡：流失率 < 5%")
    elif win_c * 100.0 / win_n > 50:
        verdict.append("[WARN] 标签极端不平衡：流失率 > 50%")

    if verdict:
        for v in verdict:
            print("  " + v)
        print("\n  → 结论倾向：**做标准流失预测模型的价值有限**")
    else:
        print("  [OK] 数据条件基本满足，可以做流失预测模型")

    print("\n  [WARN] 最终判断需要结合上面的全部诊断结果，不能只看单一指标")

    # ==========================================================
    # 导出
    # ==========================================================
    section("【导出】")
    import pandas as pd
    xlsx_path = config.TABLE_DIR / "P5可行性诊断.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("SELECT * FROM p5_last_active ORDER BY days_since_last").fetchdf().to_excel(
                writer, sheet_name="最后活跃日", index=False)
            con.execute("SELECT * FROM p5_decay").fetchdf().to_excel(
                writer, sheet_name="衰减类型", index=False)
            con.execute("SELECT * FROM p5_windows").fetchdf().to_excel(
                writer, sheet_name="滑动窗口样本", index=False)
        print(f"  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal

    section("诊断完成")
    print(f"  耗时: {time.time() - t0:.1f} 秒")
    print(f"  日志: {log_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
