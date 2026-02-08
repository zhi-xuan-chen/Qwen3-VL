"""
SimpleVisionMerger: 逐 token 线性投影，不做 patch merge。
输入 (B, N, D_vis)，输出 (B, N, D_llm)，N 任意。
"""

import torch
import torch.nn as nn


class SimpleVisionMerger(nn.Module):
    """不做 patch merge，只做逐 token 投影；N 任意，无整除要求"""

    def __init__(self, vis_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(vis_hidden_size, llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D_vis) 或 (seq_len, D_vis)，N 任意
        return self.proj(x)
