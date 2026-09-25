# -*- coding: utf-8 -*-
"""
P1-dup-test: distinguish logging duplication from genuine repeat behavior.

Tests two competing hypotheses for the 4-column repeats
(same user, item, behavior, hour):

  H1 (logging duplication): a few records repeat many times
  H2 (genuine behavior)   : many records repeat exactly twice, spread
                            evenly across users and hours

Usage:
    python src/p1_dup_test.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68
KEY4 = "user_id, item_id, behavior_type, time"


def show(df):
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))


def main():
    t0 = time.time()
    raw = f"read_csv_auto('{config.RAW_CSV.as_posix()}', header=true)"
    con = duckdb.connect()

    print(DIVIDER)
    print("  重复记录成因检验：日志重复上报 vs 真实重复行为")
    print(DIVIDER)

    # ---------- 检验 1：重复次数的分布形态 ----------
    print("\n【检验 1】重复次数的分布形态")
    print("   H1 日志重复 → 少数记录重复很多次（长尾）")
    print("   H2 真实行为 → 大量记录只重复 2 次（集中在 2）\n")

    d1 = con.execute(f"""
        WITH grp AS (
            SELECT {KEY4}, COUNT(*) AS c FROM {raw} GROUP BY 1, 2, 3, 4
        )
        SELECT c AS 重复次数,
               COUNT(*) AS 组合数,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS 占比百分比
        FROM grp
        WHERE c > 1
        GROUP BY c
        ORDER BY c
    """).fetchdf()
    show(d1.head(20))

    max_c = con.execute(f"""
        WITH grp AS (SELECT {KEY4}, COUNT(*) AS c FROM {raw} GROUP BY 1,2,3,4)
        SELECT MAX(c) FROM grp
    """).fetchone()[0]
    print(f"\n   最大重复次数: {max_c}")

    # ---------- 检验 2：重复是否集中在小部分用户 ----------
    print("\n【检验 2】重复是否集中在少数用户身上")
    print("   H1 日志重复 → 少数用户贡献绝大部分重复")
    print("   H2 真实行为 → 重复均匀分布在多数用户\n")

    d2 = con.execute(f"""
        WITH grp AS (
            SELECT {KEY4}, COUNT(*) AS c FROM {raw} GROUP BY 1, 2, 3, 4
        ),
        per_user AS (
            SELECT user_id, SUM(c - 1) AS dup_rows FROM grp GROUP BY user_id
        )
        SELECT COUNT(*) AS 有重复的用户数,
               ROUND(AVG(dup_rows), 1) AS 人均重复行数,
               MEDIAN(dup_rows) AS 中位数,
               ROUND(QUANTILE_CONT(dup_rows, 0.90), 0) AS p90,
               ROUND(QUANTILE_CONT(dup_rows, 0.99), 0) AS p99,
               MAX(dup_rows) AS 最大值
        FROM per_user WHERE dup_rows > 0
    """).fetchdf()
    show(d2)

    # ---------- 检验 3：按小时的重复率是否均匀 ----------
    print("\n【检验 3】各小时的重复率是否均匀")
    print("   H1 日志重复 → 某些小时异常高（系统问题时段）")
    print("   H2 真实行为 → 各小时重复率接近\n")

    d3 = con.execute(f"""
        WITH grp AS (
            SELECT {KEY4}, COUNT(*) AS c FROM {raw} GROUP BY 1,2,3,4
        )
        SELECT SUBSTR(time, 12, 2) AS 小时,
               SUM(c) AS 总行数,
               SUM(c - 1) AS 重复行数,
               ROUND(SUM(c - 1) * 100.0 / SUM(c), 2) AS 重复率百分比
        FROM grp
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    show(d3)

    # ---------- 检验 4：按行为类型的重复率 ----------
    print("\n【检验 4】各行为类型的重复率\n")
    d4 = con.execute(f"""
        WITH grp AS (
            SELECT {KEY4}, COUNT(*) AS c FROM {raw} GROUP BY 1,2,3,4
        )
        SELECT behavior_type AS 行为,
               CASE behavior_type WHEN 1 THEN '点击' WHEN 2 THEN '收藏'
                    WHEN 3 THEN '加购' WHEN 4 THEN '支付' END AS 名称,
               SUM(c) AS 总行数,
               SUM(c - 1) AS 重复行数,
               ROUND(SUM(c - 1) * 100.0 / SUM(c), 2) AS 重复率百分比
        FROM grp GROUP BY 1, 2 ORDER BY 1
    """).fetchdf()
    show(d4)

    # ---------- 检验 5：geohash 差异的真相 ----------
    print("\n【检验 5】4 列相同的组合里，geohash 为何不同")
    print("   检查：是否只是「一行有 geohash、其他行为 NULL」\n")

    d5 = con.execute(f"""
        WITH grp AS (
            SELECT {KEY4},
                   COUNT(*) AS n_row,
                   COUNT(user_geohash) AS n_nonnull,
                   COUNT(DISTINCT user_geohash) AS n_geo
            FROM {raw} GROUP BY 1,2,3,4
        )
        SELECT
            SUM(CASE WHEN n_row > 1 THEN 1 ELSE 0 END) AS 多行组合数,
            SUM(CASE WHEN n_row > 1 AND n_nonnull = 0 THEN 1 ELSE 0 END) AS 全为NULL的组合,
            SUM(CASE WHEN n_row > 1 AND n_nonnull = 1 THEN 1 ELSE 0 END) AS 只有一行有值的组合,
            SUM(CASE WHEN n_row > 1 AND n_nonnull > 1 THEN 1 ELSE 0 END) AS 多行都有的组合
        FROM grp
    """).fetchdf()
    show(d5)

    # ---------- 结论 ----------
    print("\n" + DIVIDER)
    print("  诊断小结")
    print(DIVIDER)
    share_2 = 0.0
    if len(d1) > 0:
        row2 = d1[d1["重复次数"] == 2]
        if len(row2) > 0:
            share_2 = float(row2.iloc[0]["占比百分比"])
    print(f"""
  最大重复次数            : {max_c}
  重复次数恰为 2 的组合占比: {share_2:.2f}%

  判断逻辑：
    数据集中「大量组合重复恰好 2 次」→ 更接近真实行为（用户看了两次）
    数据集中「少数组合重复极多次」  → 更接近日志重复上报
  """)
    print(DIVIDER)
    print(f"  耗时: {time.time() - t0:.1f} 秒")
    con.close()


if __name__ == "__main__":
    main()
