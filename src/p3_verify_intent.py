# -*- coding: utf-8 -*-
"""
P3-verify: does "low add-to-cart" simply mean users buy directly instead?

A single-chain funnel assumes click -> cart -> buy is the only route. But a
user who never touches the cart may still have bought, and in that case the
"add-to-cart" metric understates genuine intent inside the high-frequency /
low-conversion segment.

This script answers that by:
  1. Building mutually exclusive user paths (cart / fav / direct buy)
  2. Breaking total purchase events down by path
  3. Testing whether more cart behaviour leads to more purchase events

Usage:
    python src/p3_verify_intent.py
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
    log_path = config.TABLE_DIR / "P3v_运行日志.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(DIVIDER)
    print("  P3 验证：'加购少'是否等于'直接买了？'")
    print(DIVIDER)

    con = duckdb.connect(str(config.DUCKDB_PATH))

    # ==========================================================
    # 【第 1 步】用户路径互斥分类（加购 / 收藏 / 直接买 / 未买）
    # ==========================================================
    section("【第 1 步】用户路径互斥分类")
    print("  判定逻辑：一个用户归入唯一一类，按'最强的意向动作'排序")
    print("           有加购 > 仅收藏 > 直接购买 > 未购买\n")

    con.execute("DROP TABLE IF EXISTS ads_user_path")
    con.execute("""
        CREATE TABLE ads_user_path AS
        WITH u AS (
            SELECT user_id,
                   MAX(has_cart) AS c,
                   MAX(has_fav)  AS f,
                   MAX(has_buy)  AS b
            FROM dws_user_daily GROUP BY user_id
        )
        SELECT
            user_id, c, f, b,
            CASE
                WHEN b = 0 THEN 'D-未购买'
                WHEN b = 1 AND c = 1 THEN 'A-加购后支付'
                WHEN b = 1 AND c = 0 AND f = 1 THEN 'B-仅收藏后支付'
                ELSE 'C-直接支付（无加购无收藏）'
            END AS path
        FROM u
    """)

    print("  路径分布：")
    show(con.execute("""
        SELECT path AS 用户路径, COUNT(*) AS 用户数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占全部用户百分比,
               ROUND(COUNT(*) * 100.0 /
                     NULLIF((SELECT COUNT(*) FROM ads_user_path WHERE b = 1), 0), 2)
                     AS 占购买用户百分比
        FROM ads_user_path GROUP BY 1 ORDER BY 1
    """).fetchdf())

    # ==========================================================
    # 【第 2 步】购买事件按路径拆解
    # ==========================================================
    section("【第 2 步】购买事件按路径拆解")
    print("  关键区分：人数 vs 事件数")
    print("            '加购人数' 与 '加购次数' 是两回事，不能混用\n")

    print("  各路径用户贡献的购买事件数：")
    show(con.execute("""
        SELECT p.path AS 用户路径,
               COUNT(DISTINCT p.user_id) AS 用户数,
               SUM(d.buy_pv)             AS 购买事件数,
               ROUND(SUM(d.buy_pv) * 100.0 /
                     (SELECT SUM(buy_pv) FROM dws_user_daily), 2) AS 占购买事件百分比,
               ROUND(AVG(s.total_buy), 1) AS 人均购买次数
        FROM ads_user_path p
        JOIN dws_user_daily d ON p.user_id = d.user_id
        JOIN ads_user_base  s ON p.user_id = s.user_id
        GROUP BY p.path ORDER BY p.path
    """).fetchdf())

    # ==========================================================
    # 【第 3 步】直接购买到底占多少
    # ==========================================================
    section("【第 3 步】核心问题：'加购少'的用户是否直接买了？")
    print("  对比三种口径下'直接购买'的规模\n")

    show(con.execute("""
        WITH all_u AS (SELECT COUNT(*) AS total FROM ads_user_path),
             buyers AS (SELECT COUNT(*) AS n FROM ads_user_path WHERE b = 1),
             direct AS (SELECT COUNT(*) AS n FROM ads_user_path WHERE path LIKE 'C-%'),
             no_cart AS (SELECT COUNT(*) AS n FROM ads_user_path WHERE c = 0 AND b = 1),
             no_cart_fav AS (SELECT COUNT(*) AS n FROM ads_user_path
                             WHERE c = 0 AND f = 0 AND b = 1)
        SELECT
            (SELECT total FROM all_u)     AS 全部用户,
            (SELECT n FROM buyers)        AS 购买用户,
            (SELECT n FROM no_cart)       AS 购买但未加购,
            ROUND((SELECT n FROM no_cart) * 100.0
                  / (SELECT n FROM buyers), 2) AS 未加购占购买用户百分比,
            (SELECT n FROM no_cart_fav)   AS 购买但未加购未收藏,
            ROUND((SELECT n FROM no_cart_fav) * 100.0
                  / (SELECT n FROM buyers), 2) AS 完全直接购买占比
    """).fetchdf())

    print("     若占比高 → 该疑点成立，加购率会低估真实购买意向\n")

    # ==========================================================
    # 【第 4 步】分人群看直接购买的占比
    # ==========================================================
    section("【第 4 步】各人群的路径构成 —— 高频低转化用户是否特别倾向直接买？")

    show(con.execute("""
        SELECT b.segment AS 用户分层,
               COUNT(*) AS 用户数,
               SUM(CASE WHEN p.path = 'A-加购后支付' THEN 1 ELSE 0 END) AS 加购后支付,
               SUM(CASE WHEN p.path = 'B-仅收藏后支付' THEN 1 ELSE 0 END) AS 仅收藏后支付,
               SUM(CASE WHEN p.path = 'C-直接支付（无加购无收藏）' THEN 1 ELSE 0 END) AS 直接支付,
               SUM(CASE WHEN p.path = 'D-未购买' THEN 1 ELSE 0 END) AS 未购买,
               ROUND(SUM(CASE WHEN p.path = 'C-直接支付（无加购无收藏）' THEN 1 ELSE 0 END)
                     * 100.0 / NULLIF(SUM(CASE WHEN p.b = 1 THEN 1 ELSE 0 END), 0), 2)
                     AS 直接支付占购买百分比
        FROM ads_user_path p
        JOIN ads_user_base b ON p.user_id = b.user_id
        GROUP BY b.segment ORDER BY b.segment
    """).fetchdf(), max_rows=20)

    # ==========================================================
    # 【第 5 步】加购次数与购买次数的关系（验证"加购是否有效信号"）
    # ==========================================================
    section("【第 5 步】加购次数 → 购买次数，是同向关系吗？")
    print("  方法：把用户按加购次数分档，看平均购买次数是否递增")
    print("        若递增 → 加购确实是购买的有效前兆信号\n")

    show(con.execute("""
        SELECT
            CASE
                WHEN total_cart = 0              THEN '1-零加购'
                WHEN total_cart <= 10            THEN '2-低加购(1-10)'
                WHEN total_cart <= 30            THEN '3-中加购(11-30)'
                WHEN total_cart <= 90            THEN '4-高加购(31-90)'
                ELSE                                  '5-极高加购(>90)'
            END AS 加购次数分档,
            COUNT(*)                     AS 用户数,
            ROUND(AVG(total_click), 0)   AS 平均点击,
            ROUND(AVG(total_cart), 1)    AS 平均加购,
            ROUND(AVG(total_buy), 1)     AS 平均购买,
            ROUND(SUM(total_buy) * 100.0
                  / NULLIF(SUM(total_click), 0), 3) AS 点击转支付率百分比
        FROM ads_user_base
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    # ==========================================================
    # 【第 6 步】按品类集中度分组（验证"扩大品类"是否有效）
    # ==========================================================
    section("【第 6 步】浏览品类多的人，买得更多吗？")
    print("  验证一个反向可能：'扩大品类或许增加购买，但也可能让用户厌烦'")
    print("  方法：把用户按浏览品类数分档，看平均购买次数\n")

    show(con.execute("""
        WITH uc AS (
            SELECT d.user_id, COUNT(DISTINCT d.item_category) AS cat_cnt
            FROM dwd_user_behavior d GROUP BY d.user_id
        )
        SELECT
            CASE
                WHEN uc.cat_cnt <= 20            THEN '1-极窄(<=20)'
                WHEN uc.cat_cnt <= 60            THEN '2-较窄(21-60)'
                WHEN uc.cat_cnt <= 120           THEN '3-中等(61-120)'
                WHEN uc.cat_cnt <= 250           THEN '4-较宽(121-250)'
                ELSE                                  '5-极宽(>250)'
            END AS 浏览品类数分档,
            COUNT(*)                          AS 用户数,
            ROUND(AVG(uc.cat_cnt), 0)         AS 平均品类数,
            ROUND(AVG(b.active_days), 1)      AS 平均活跃天数,
            ROUND(AVG(b.total_buy), 1)        AS 平均购买次数,
            ROUND(AVG(b.total_buy) * 1.0
                  / NULLIF(AVG(b.active_days), 0), 3) AS 日均购买次数
        FROM uc JOIN ads_user_base b ON uc.user_id = b.user_id
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("\n  ※ 看【日均购买次数】这一列：")
    print("     若随品类数递增 → 浏览越广，购买效率越高")
    print("     若先增后降       → 存在最优品类数，过度扩品类会稀释转化\n")

    # ==========================================================
    # 导出
    # ==========================================================
    section("【导出】")
    import pandas as pd
    xlsx_path = config.TABLE_DIR / "P3v_验证结果.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            con.execute("""
                SELECT path AS 用户路径, COUNT(*) AS 用户数,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占全部用户百分比
                FROM ads_user_path GROUP BY 1 ORDER BY 1
            """).fetchdf().to_excel(writer, sheet_name="路径分布", index=False)
            con.execute("""
                SELECT b.segment AS 用户分层, COUNT(*) AS 用户数,
                       SUM(CASE WHEN p.path = 'A-加购后支付' THEN 1 ELSE 0 END) AS 加购后支付,
                       SUM(CASE WHEN p.path = 'B-仅收藏后支付' THEN 1 ELSE 0 END) AS 仅收藏后支付,
                       SUM(CASE WHEN p.path = 'C-直接支付（无加购无收藏）' THEN 1 ELSE 0 END) AS 直接支付,
                       SUM(CASE WHEN p.path = 'D-未购买' THEN 1 ELSE 0 END) AS 未购买
                FROM ads_user_path p JOIN ads_user_base b ON p.user_id = b.user_id
                GROUP BY b.segment ORDER BY b.segment
            """).fetchdf().to_excel(writer, sheet_name="各人群路径", index=False)
        print(f"  已导出: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] Excel 导出失败: {type(e).__name__}: {e}")

    con.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal

    section("验证完成")
    print(f"  耗时: {time.time() - t0:.1f} 秒")
    print(f"  日志: {log_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
