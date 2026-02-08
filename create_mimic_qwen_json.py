#!/usr/bin/env python3
"""
将 MIMIC-CXR 数据集的标注 CSV 转换为 Qwen VL 训练所需的 JSON 格式。

数据映射：
- human 输入：image_paths 中的图像 + 固定 prompt（让模型根据图像生成报告）
- gpt 回答：findings_section（原始 CSV 中的 findings 部分）

用法示例：
    python create_mimic_qwen_json.py \
        --csv_path /jhcnas4/XR/MIMIC-CXR/labels/mimic_engineered_merge_with_split.csv \
        --image_base_path /jhcnas4/XR/MIMIC-CXR/imgs-1024 \
        --output_path mimic_qwen_train.json \
        --split train
"""

import argparse
import ast
import json
import os
from typing import List, Optional


# 默认固定 prompt：让模型根据胸部 X 光片生成放射学报告 findings 部分
DEFAULT_PROMPT = "Based on the chest X-ray image(s) provided, please generate the radiology report findings section."


def parse_image_paths(image_paths_str) -> List[str]:
    """解析 image_paths 列，可能是 Python list 字符串。"""
    if image_paths_str is None or (
        isinstance(image_paths_str, float) and str(image_paths_str) == "nan"
    ):
        return []
    s = str(image_paths_str).strip()
    try:
        paths = ast.literal_eval(s)
        if isinstance(paths, list):
            return [str(p).strip() for p in paths if p]
        return [s]
    except (ValueError, SyntaxError):
        return [p.strip() for p in s.split(",") if p.strip()]


def build_human_value(image_count: int, prompt: str) -> str:
    """构建 human 的 value：N 个 <image> 标签 + prompt。"""
    tags = "\n".join(["<image>"] * image_count)
    return f"{tags}\n{prompt}"


def row_to_qwen_item(
    row: dict,
    image_base_path: str,
    prompt: str,
    use_relative_path: bool = False,
) -> Optional[dict]:
    """
    将 CSV 一行转换为 Qwen JSON 单条。

    Returns:
        Qwen 格式的 dict，或 None（若数据无效则跳过）
    """
    image_paths_str = row.get("image_paths", "")
    findings = row.get("findings_section", "")

    paths = parse_image_paths(image_paths_str)
    if not paths:
        return None  # skip_reason: "no_image_paths"

    findings = str(findings).strip() if findings is not None and str(findings) != "nan" else ""
    if not findings:
        return None  # skip_reason: "no_findings"

    if use_relative_path:
        full_paths = paths
    else:
        full_paths = [
            os.path.join(image_base_path, p) if image_base_path else p
            for p in paths
        ]

    if len(full_paths) == 1:
        image_field = full_paths[0]
    else:
        image_field = full_paths

    human_value = build_human_value(len(paths), prompt)

    return {
        "image": image_field,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": findings},
        ],
    }


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="将 MIMIC-CXR CSV 转为 Qwen JSON")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/jhcnas4/XR/MIMIC-CXR/labels/mimic_engineered_merge_with_split.csv",
        help="MIMIC 标注 CSV 路径",
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default="/jhcnas4/XR/MIMIC-CXR/imgs-1024",
        help="图像根目录，会与该列中的相对路径拼接",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mimic_qwen_train.json",
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="只处理指定 split（train/val/test），不指定则处理全部",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="固定 prompt，用于让模型根据图像生成报告",
    )
    parser.add_argument(
        "--use_relative_path",
        action="store_true",
        help="输出中使用相对路径（配合 data_path 使用），否则使用绝对路径",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理多少条（用于快速测试）",
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.csv_path,
        low_memory=False,
        dtype={"findings_section": str},  # 强制按字符串读取，避免 mixed types 警告
    )
    if args.split:
        df = df[df["split"] == args.split]

    if args.max_samples:
        df = df.head(args.max_samples)

    result = []
    skip_no_paths = 0
    skip_no_findings = 0
    for _, row in df.iterrows():
        paths = parse_image_paths(row.get("image_paths", ""))
        findings = row.get("findings_section", "")
        findings_ok = (
            findings is not None
            and str(findings) != "nan"
            and str(findings).strip()
        )

        if not paths:
            skip_no_paths += 1
        elif not findings_ok:
            skip_no_findings += 1
        else:
            item = row_to_qwen_item(
                row.to_dict(),
                image_base_path=args.image_base_path,
                prompt=args.prompt,
                use_relative_path=args.use_relative_path,
            )
            if item is not None:
                result.append(item)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    skipped = skip_no_paths + skip_no_findings
    print(f"转换完成: 有效 {len(result)} 条，跳过 {skipped} 条，已写入 {args.output_path}")
    if skipped:
        print(f"  跳过原因: image_paths 为空={skip_no_paths}, findings_section 为空={skip_no_findings}")


if __name__ == "__main__":
    main()
