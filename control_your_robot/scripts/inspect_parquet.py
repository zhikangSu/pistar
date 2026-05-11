#!/usr/bin/env python
"""
查看 LeRobot Parquet 文件的格式和内容

使用方法:
python scripts/inspect_parquet.py --file /path/to/episode_000000.parquet
"""

import argparse
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

VALUE_LABEL_COLUMNS = ("value_lable", "value")


def inspect_parquet(file_path, num_rows=5):
    """
    查看 parquet 文件的详细信息

    Args:
        file_path: parquet 文件路径
        num_rows: 显示的行数
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print("=" * 80)
    print(f"文件: {file_path}")
    print("=" * 80)

    # 1. 使用 pyarrow 读取元数据
    print("\n📊 文件元数据:")
    print("-" * 80)

    parquet_file = pq.ParquetFile(file_path)

    print(f"行数: {parquet_file.metadata.num_rows}")
    print(f"列数: {parquet_file.metadata.num_columns}")
    print(f"文件大小: {file_path.stat().st_size / 1024:.2f} KB")
    print(f"行组数: {parquet_file.metadata.num_row_groups}")

    # 2. Schema 信息
    print("\n📋 Schema (列信息):")
    print("-" * 80)

    schema = parquet_file.schema_arrow
    for i, field in enumerate(schema):
        print(f"{i+1}. {field.name:30s} - {field.type}")

    # 3. 使用 pandas 读取数据
    print(f"\n📦 数据预览 (前 {num_rows} 行):")
    print("-" * 80)

    df = pd.read_parquet(file_path)

    # 显示每列的详细信息
    for col in df.columns:
        print(f"\n列名: {col}")
        print(f"  类型: {df[col].dtype}")
        print(f"  形状: {df[col].shape}")

        # 显示前几个值
        if num_rows > 0:
            print(f"  前 {min(num_rows, len(df))} 个值:")
            for idx in range(min(num_rows, len(df))):
                value = df[col].iloc[idx]
                if hasattr(value, 'shape'):
                    print(f"    [{idx}] shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    [{idx}] {value}")

    # 4. 统计信息
    print("\n📈 统计信息:")
    print("-" * 80)

    print(f"\n总帧数: {len(df)}")

    # 检查是否有 episode_index
    if 'episode_index' in df.columns:
        print(f"Episode Index: {df['episode_index'].iloc[0] if len(df) > 0 else 'N/A'}")

    # 检查是否有 timestamp
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values
        if len(timestamps) > 1:
            intervals = timestamps[1:] - timestamps[:-1]
            print(f"时间间隔 (平均): {intervals.mean():.4f} 秒")
            print(f"时间间隔 (最小): {intervals.min():.4f} 秒")
            print(f"时间间隔 (最大): {intervals.max():.4f} 秒")

    # 检查是否有 intervention (RL 数据集)
    if 'intervention' in df.columns:
        intervention_count = df['intervention'].sum()
        print(f"\n🎮 强化学习信息:")
        print(f"  干预帧数: {intervention_count} / {len(df)} ({intervention_count/len(df)*100:.1f}%)")
        print(f"  自主帧数: {len(df) - intervention_count} / {len(df)} ({(len(df)-intervention_count)/len(df)*100:.1f}%)")

    value_col = next((col for col in VALUE_LABEL_COLUMNS if col in df.columns), None)
    if value_col is not None:
        print(f"  {value_col} 范围: [{df[value_col].min():.2f}, {df[value_col].max():.2f}]")
        print(f"  最后一帧 {value_col}: {df[value_col].iloc[-1]:.2f}")

    print("\n" + "=" * 80)


def compare_episodes(file1, file2):
    """比较两个 episode 的差异"""
    print("=" * 80)
    print("比较两个 Episodes")
    print("=" * 80)

    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    print(f"\nEpisode 1: {Path(file1).name}")
    print(f"  帧数: {len(df1)}")
    print(f"  列: {list(df1.columns)}")

    print(f"\nEpisode 2: {Path(file2).name}")
    print(f"  帧数: {len(df2)}")
    print(f"  列: {list(df2.columns)}")

    # 比较 schema
    print(f"\n列差异:")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    if cols1 == cols2:
        print("  ✓ 两个文件的列完全相同")
    else:
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1
        if only_in_1:
            print(f"  仅在 Episode 1: {only_in_1}")
        if only_in_2:
            print(f"  仅在 Episode 2: {only_in_2}")


def main():
    parser = argparse.ArgumentParser(description="查看 LeRobot Parquet 文件格式")
    parser.add_argument("--file", type=str, required=True, help="Parquet 文件路径")
    parser.add_argument("--rows", type=int, default=5, help="显示的行数 (默认: 5)")
    parser.add_argument("--compare", type=str, help="比较另一个 parquet 文件")

    args = parser.parse_args()

    inspect_parquet(args.file, args.rows)

    if args.compare:
        print("\n")
        compare_episodes(args.file, args.compare)


if __name__ == "__main__":
    main()
