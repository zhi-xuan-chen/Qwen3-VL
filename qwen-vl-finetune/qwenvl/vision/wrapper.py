"""
CustomVisionWrapper: 封装自定义视觉 encoder + SimpleVisionMerger，对接 Qwen 训练框架。
图片在数据管道中已经由 get_custom_image_transform 按 encoder 要求处理好，
此处直接调 extract_fn 提取特征 → merger 投影 → 输出 (total_tokens, D_llm)。

重要：
- spatial_merge_size 需与数据管道中 processor.image_processor.merge_size 保持一致
  （默认 2），用于 get_image_features 中计算 split_sizes。
- forward 返回 (flat_embeds, [])，其中 deepstack 为空列表（自定义 encoder 无多层 ViT 输出）。
  外部 get_image_features 会在拿到 flat_embeds 后按 image_grid_thw 切分为 per-image tuple。
"""

from typing import Optional

import torch
import torch.nn as nn

from .merger import SimpleVisionMerger
from .utils import (
    VIS_ENCODER_CONFIG,
    create_vision_encoder,
    get_extract_features_fn,
)


class CustomVisionWrapper(nn.Module):
    """
    将 qwenvl.vision 中的 encoder（gloria, chexzero, rad_dino 等）封装为
    可与 Qwen model.visual 替换的模块。
    包含 .merger 子模块，输出格式与 downstream 兼容。
    """

    def __init__(
        self,
        encoder_cfg: dict,
        llm_hidden_size: int,
        spatial_merge_size: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        model_name = encoder_cfg.get("model_name", "rad_dino").lower()
        if model_name not in VIS_ENCODER_CONFIG:
            raise ValueError(
                f"Unknown encoder '{model_name}'. Supported: {list(VIS_ENCODER_CONFIG.keys())}"
            )
        cfg = VIS_ENCODER_CONFIG[model_name]
        vis_dim = cfg["vis_dim"]

        self.encoder = create_vision_encoder(encoder_cfg)
        self.extract_fn = get_extract_features_fn(model_name)
        self.merger = SimpleVisionMerger(
            vis_hidden_size=vis_dim,
            llm_hidden_size=llm_hidden_size,
        )
        self.model_name = model_name
        self.input_size = cfg["input_size"]
        self.channels = cfg["channels"]
        self._device = device
        # 必须与数据管道的 merge_size 一致，get_image_features 用它算 split_sizes
        self.spatial_merge_size = spatial_merge_size

    @property
    def device(self) -> torch.device:
        if self._device is not None:
            return self._device
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Qwen get_image_features 会访问 self.visual.dtype，需与 nn.Module 行为一致。"""
        return next(self.parameters()).dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple:
        """
        Qwen visual 接口：接收 pixel_values, grid_thw，返回 (image_embeds, deepstack_image_embeds)。
        图片已在数据管道中按 encoder 要求处理好，此处直接提取特征并投影。

        与原始 Qwen3VLVisionModel.forward 保持一致的返回格式：
            - image_embeds: (total_tokens, D_llm) 所有图片的 token 拼接在一起
            - deepstack_image_embeds: list（自定义 encoder 无 deepstack，返回空列表）
        外部 get_image_features 会用 image_grid_thw 和 spatial_merge_size 将
        image_embeds 切分为 per-image 的 tuple。

        Args:
            pixel_values: (B, C, H, W) 已经过 get_custom_image_transform 处理的 tensor
            grid_thw: (num_images, 3) 每张图的 grid (t,h,w)（本 wrapper 不使用）

        Returns:
            (image_embeds, deepstack_image_embeds) 两元素 tuple
        """
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        # Qwen 官方视觉编码器只输出 patch tokens，无 class token；此处保持一致，仅用 patch_tokens
        patch_tokens, _ = self.extract_fn(self.encoder, pixel_values)  # (B, N, D_vis)
        image_embeds = self.merger(patch_tokens)                       # (B, N, D_llm)
        flat = image_embeds.reshape(-1, image_embeds.size(-1))          # (total_tokens, D_llm)

        # 自定义 encoder 无 deepstack 多层特征，返回空列表
        # 切分 per-image 的逻辑交给 get_image_features 处理
        return (flat, [])

    def print_trainable_parameters(self) -> None:
        """兼容 Qwen3VLVisionModel 的接口。"""
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        merger_trainable = sum(p.numel() for p in self.merger.parameters() if p.requires_grad)
        print(f"CustomVisionWrapper - Encoder trainable: {enc_trainable}, Merger trainable: {merger_trainable}")


def build_custom_vision_for_qwen(
    encoder_cfg: dict,
    llm_hidden_size: int,
    spatial_merge_size: int = 2,
    device: Optional[torch.device] = None,
) -> CustomVisionWrapper:
    """构建可替换 model.visual 的 CustomVisionWrapper。"""
    return CustomVisionWrapper(
        encoder_cfg=encoder_cfg,
        llm_hidden_size=llm_hidden_size,
        spatial_merge_size=spatial_merge_size,
        device=device,
    )
