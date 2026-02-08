import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch import nn
from transformers import AutoImageProcessor
from transformers import AutoModel
from transformers.feature_extraction_utils import BatchFeature


__version__ = "0.1.0"

TypeClsToken = Float[Tensor, "batch_size embed_dim"]
TypePatchTokensFlat = Float[Tensor, "batch_size (height width) embed_dim"]
TypePatchTokens = Float[Tensor, "batch_size embed_dim height width"]
TypeInputImages = Image.Image | list[Image.Image]


class RadDino(nn.Module):
    _REPO = "microsoft/rad-dino"

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(self._REPO)
        self.processor = AutoImageProcessor.from_pretrained(
            self._REPO, use_fast=False, do_rescale=False
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def preprocess(self, image_or_images: TypeInputImages) -> BatchFeature:
        return self.processor(image_or_images, return_tensors="pt")

    def encode(
        self, inputs: BatchFeature | Tensor
    ) -> tuple[TypeClsToken, TypePatchTokensFlat]:
        # Accept either a preprocessed BatchFeature or a tensor of pixel_values
        if isinstance(inputs, BatchFeature):
            batch = inputs.to(self.device)
        else:
            # Assume tensor of shape (B, C, H, W)
            batch = {"pixel_values": inputs.to(self.device)}

        outputs = self.model(**batch)
        cls_token = outputs.last_hidden_state[:, 0]
        patch_tokens = outputs.last_hidden_state[:, 1:]
        return cls_token, patch_tokens

    def reshape_patch_tokens(
        self,
        patch_tokens_flat: TypePatchTokensFlat,
    ) -> TypePatchTokens:
        input_size = self.processor.crop_size["height"]
        patch_size = self.model.config.patch_size
        embeddings_size = input_size // patch_size
        patches_grid = rearrange(
            patch_tokens_flat,
            "batch (height width) embed_dim -> batch embed_dim height width",
            height=embeddings_size,
        )
        return patches_grid

    @torch.inference_mode()
    def extract_features(
        self,
        image_or_images: TypeInputImages,
    ) -> tuple[TypeClsToken, TypePatchTokens]:
        batch = self.preprocess(image_or_images).to(self.device)
        cls_token, patch_tokens_flat = self.encode(batch)
        patch_tokens = self.reshape_patch_tokens(patch_tokens_flat)
        return cls_token, patch_tokens

    def extract_cls_token(self, image_or_images: TypeInputImages) -> TypeClsToken:
        cls_token, _ = self.extract_features(image_or_images)
        return cls_token

    def extract_patch_tokens(self, image_or_images: TypeInputImages) -> TypePatchTokens:
        _, patch_tokens = self.extract_features(image_or_images)
        return patch_tokens

    def forward(self, *args) -> tuple[TypeClsToken, TypePatchTokens]:
        return self.extract_features(*args)