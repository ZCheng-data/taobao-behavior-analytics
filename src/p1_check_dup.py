# -*- coding: utf-8 -*-
"""
P1-fix: duplicate count cross-validation.

The step-5 diagnostic and the step-10 Parquet row count disagreed
(6.2M vs 8.2M). This script runs four independent counts so the
real deduplication result can be determined empirically.

Usage:
    python src/p1_check_dup.py
"""
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIVIDER = "=" * 68

KEY4 = "user_id, item_id, behavior_type, time"


def main():
    t0 = time.time()
    raw = f"read_csv_auto('{config.RAW_CSV.as_posix()}', header=true)"

    con = duckdb.connect()
    print(DIVIDER)
    print("  重复记录口径交叉验证")
    print(DIVIDER)

    # ---- 基础规模 ----
    total = con.execute(f"SELECT COUNT(*) FROM {raw}").fetchone()[0]
    print(f"\n  原始总行数: {total:,}\n")

    # ---- 口径 A：4 列 GROUP BY ----
    print("  [A] GROUP BY (user_id, item_id, behavior_type, time)")
    a = con.execute(f"""
        SELECT COUNT(*) AS 唯一组合数,
               SUM(c) AS 总行数,
               SUM(c - 1) AS 重复行数,
               COUNT(*) FILTER (WHERE c > 1) AS 有重复的组合数
        FROM (SELECT {KEY4}, COUNT(*) AS c FROM {raw} GROUP BY 1, 2, 3, 4)
    """).fetchdf()
    print(a.to_string(index=False))
    a_uniq = int(a.loc[0, "唯一组合数"])
    a_dup = int(a.loc[0, "重复行数"])

    # ---- 口径 B：4 列 DISTINCT ----
    print("\n  [B] SELECT DISTINCT (user_id, item_id, behavior_type, time)")
    b_uniq = con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT {KEY4} FROM {raw})"
    ).fetchone()[0]
    print(f"      唯一组合数: {b_uniq:,}")

    # ---- 口径 C：10 列 DISTINCT（与第 10 步落盘口径一致） ----
    print("\n  [C] SELECT DISTINCT * —— 全字段去重")
    c_uniq = con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {raw})"
    ).fetchone()[0]
    print(f"      唯一行数: {c_uniq:,}")

    # ---- 口径 D：4 列相同但其他字段不同 ----
    print("\n  [D] 是否存在 (user,item,behavior,time) 相同、但其他字段不同的记录？")
    d = con.execute(f"""
        SELECT
            COUNT(*) AS 组合数,
            SUM(CASE WHEN n_cat > 1 THEN 1 ELSE 0 END) AS 类目不同的组合数,
            SUM(CASE WHEN n_geo > 1 THEN 1 ELSE 0 END) AS 地理不同的组合数,
            SUM(CASE WHEN n_row > 1 THEN 1 ELSE 0 END) AS 多行组合数
        FROM (
            SELECT {KEY4},
                   COUNT(DISTINCT item_category) AS n_cat,
                   COUNT(DISTINCT user_geohash)  AS n_geo,
                   COUNT(*)                      AS n_row
            FROM {raw} GROUP BY 1, 2, 3, 4
        )
    """).fetchdf()
    print(d.to_string(index=False))

    # ---- 结论 ----
    print("\n" + DIVIDER)
    print("  诊断结论")
    print(DIVIDER)
    print(f"""
  口径 A（GROUP BY 4 列）  : {a_uniq:>12,}
  口径 B（DISTINCT 4 列）  : {b_uniq:>12,}
  口径 C（DISTINCT 全字段）: {c_uniq:>12,}

  原始总行数               : {total:>12,}
  """)

    if abs(a_uniq - b_uniq) <= 1 and abs(b_uniq - c_uniq) <= 1:
        print("  → 三个口径一致，第 5 步结果可信。")
    elif abs(a_uniq - c_uniq) > 1000:
        print("  → 口径 A/B 与口径 C 不一致。")
        print(f"    差异 {abs(a_uniq - c_uniq):,} 行，"
              f"说明存在「4 列相同但其他字段不同」的记录（见口径 D）。")
    else:
        print("  → 结果基本一致，存在少量边界差异。")

    print(f"\n  真实重复行数（以全字段去重为准）: {total - c_uniq:,}")
    print(f"  真实重复率                      : "
          f"{(total - c_uniq) * 100.0 / total:.4f}%")
    print(f"\n  耗时: {time.time() - t0:.1f} 秒")
    print(DIVIDER)
    con.close()


if __name__ == "__main__":
    main()
