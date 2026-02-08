import torch
import torch.nn as nn

# 各视觉 encoder 的输出维度与输入尺寸（用于 SimpleVisionMerger 和预处理）
# num_tokens：仅 patch 数量，与 Qwen 官方视觉编码器一致（无 class token）
VIS_ENCODER_CONFIG = {
    "gloria":       {"vis_dim": 1024, "input_size": 224, "channels": 3, "num_tokens": 49},    # patch only
    "chexzero":     {"vis_dim": 768,  "input_size": 224, "channels": 3, "num_tokens": 49},    # patch only
    "maco":         {"vis_dim": 768,  "input_size": 224, "channels": 3, "num_tokens": 196},   # patch only
    "rad_dino":     {"vis_dim": 768,  "input_size": 518, "channels": 3, "num_tokens": 1369},  # patch only
    "ark_base":     {"vis_dim": 1024, "input_size": 224, "channels": 3, "num_tokens": 49},    # patch only
    "ark_large":    {"vis_dim": 1536, "input_size": 768, "channels": 3, "num_tokens": 576},   # patch only
    "our_ark_base": {"vis_dim": 1024, "input_size": 224, "channels": 3, "num_tokens": 49},    # patch only
    "our_ark_large":{"vis_dim": 1536, "input_size": 768, "channels": 3, "num_tokens": 576},   # patch only
    "chess":        {"vis_dim": 2048, "input_size": 512, "channels": 1, "num_tokens": 256},   # patch only
}


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


def create_vision_encoder(cfg):
    """Create encoder model from config."""
    model_name = cfg["model_name"]

    # DenseNet121
    if model_name.lower() == "gloria":
        from torchvision import models as models_2d

        encoder = models_2d.densenet121(weights=None)
        encoder.classifier = Identity()
        weight = torch.load(
            "/jhcnas4/XR/pretrained_weights/Gloria.ckpt", 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        weight = weight["state_dict"]
        state_dict = {
            k.replace("gloria.img_encoder.model.", ""): v
            for k, v in weight.items()
            if "gloria.img_encoder.model" in k
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=True)

    # ViT
    elif model_name.lower() == "chexzero":
        from .clip_vit import VisualTransformer

        encoder = VisualTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=8,
            output_dim=512,
        )
        weight = torch.load(
            "/jhcnas4/XR/pretrained_weights/CheXzero.pth", 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        state_dict = {
            k.replace("visual.", ""): v 
            for k, v in weight.items() 
            if "visual." in k
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=True)

    # ViT
    elif model_name.lower() == "maco":
        from timm.models.vision_transformer import VisionTransformer
        encoder = VisionTransformer()
        encoder.head = Identity()
        weight = torch.load(
            "/jhcnas4/XR/pretrained_weights/MaCo.pth",
            map_location="cpu",
            weights_only=False,
        )
        state_dict = weight["model"]
        # Maco weight includes language model weights
        # so we need to load them with strict=False
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 178

    # ViT
    elif model_name.lower() == "rad_dino":
        from .rad_dino import RadDino

        encoder = RadDino()
        
        if cfg.get("pretrained_path", None) is None:
            return encoder
        
        print(f"Loading pretrained weights from {cfg['pretrained_path']}")
        weight = torch.load(
            cfg["pretrained_path"], 
            map_location="cpu",
            weights_only=False
        )
        state_dict = weight["state_dict"]
        state_dict = {
            k.replace("model.vision_encoder.", ""): v
            for k, v in state_dict.items()
            if "vision_encoder." in k
        }

        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)

    # Swin Transformer
    elif model_name.lower() == "ark_base":
        from timm.models.swin_transformer import SwinTransformer

        encoder = SwinTransformer(
            img_size=224,
            patch_size=4,
            window_size=7,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
        )
        encoder.head = Identity()

        pretrained_path = "/jhcnas4/XR/pretrained_weights/ark6_swinbase_224_ep200.pth.tar"
        print(f"Loading pretrained weights from {pretrained_path}")
        weight = torch.load(
            pretrained_path, 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        state_dict = weight["teacher"]
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 16

    # Swin Transformer
    elif model_name.lower() == "ark_large":
        from timm.models.swin_transformer import SwinTransformer

        encoder = SwinTransformer(
            img_size=768,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
        )
        encoder.head = Identity()

        pretrained_path = "/jhcnas4/XR/pretrained_weights/Ark6_swinLarge768_ep50.pth.tar"
        print(f"Loading pretrained weights from {pretrained_path}")
        weight = torch.load(
            pretrained_path, 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        state_dict = weight["teacher"]
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 16

    elif model_name.lower() == "our_ark_base":
        from timm.models.swin_transformer import SwinTransformer

        encoder = SwinTransformer(
            img_size=224,
            patch_size=4,
            window_size=7,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
        )
        encoder.head = Identity()

        if cfg.get("pretrained_path", None) is None:
            raise ValueError("pretrained_path must be specified for our_ark_base model")
        pretrained_path = cfg["pretrained_path"]
        print(f"Loading pretrained weights from {pretrained_path}")
        weight = torch.load(
            pretrained_path, 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        state_dict = weight["state_dict"]
        state_dict = {
            k.replace("model.", "").replace("vision_encoder.", ""): v
            for k, v in state_dict.items()
            if ("vision_encoder" in k)
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)

    elif model_name.lower() == "our_ark_large":
        from timm.models.swin_transformer import SwinTransformer

        encoder = SwinTransformer(
            img_size=768,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
        )
        encoder.head = Identity()

        if cfg.get("pretrained_path", None) is None:
            raise ValueError("pretrained_path must be specified for our_ark_large model")
        pretrained_path = cfg["pretrained_path"]
        print(f"Loading pretrained weights from {pretrained_path}")
        weight = torch.load(
            pretrained_path, 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        state_dict = weight["state_dict"]
        state_dict = {
            k.replace("model.", "").replace("vision_encoder.", ""): v
            for k, v in state_dict.items()
            if ("vision_encoder" in k)
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)

    # ResNet50
    elif model_name.lower() == "chess":
        from .resnet import resnet50
        encoder = resnet50()

        pretrained_model = "/jhcnas4/XR/pretrained_weights/CheSS.pth.tar"
        weight = torch.load(
            pretrained_model, 
            map_location="cpu",
            weights_only=False
        )

        # rename moco pre-trained keys
        state_dict = weight["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        encoder.fc = Identity()
        missing, unexpected = encoder.load_state_dict(state_dict, strict=True)

    print("missing:", missing)
    print("unexpected:", len(unexpected))
    assert len(missing) == 0
    return encoder


def extract_swin_features(encoder, images):
    """
    Extract features from ARK models (Swin Transformer based).
    ark_base, ark_large, our_ark_base, our_ark_large
    
    Returns:
        patch_tokens: [B, H*W, C]
        cls_token: [B, C]
    """
    # ARK uses timm SwinTransformer
    # forward_features pools to [B, C], so we need to manually extract before pooling
    # to get patch tokens [B, L, C] where L is number of tokens
    x = encoder.patch_embed(images)
    if encoder.absolute_pos_embed is not None:
        x = x + encoder.absolute_pos_embed
    x = encoder.pos_drop(x)
    x = encoder.layers(x)
    x = encoder.norm(x)  # [B, L, C] where L is number of tokens
    patch_tokens = x
    cls_token = encoder.avgpool(x.transpose(1, 2))  # [B, C, 1, 1]
    cls_token = torch.flatten(cls_token, 1)  # [B, C]
    
    return patch_tokens, cls_token

def extract_vit_features(encoder, images):
    """
    Extract patch tokens from timm VisionTransformer.
    
    Returns:
        patch_tokens: [B, H*W, C]
    """
    # Manual extraction for timm VisionTransformer
    # patch_embed -> add CLS -> blocks -> norm -> extract patch tokens
    x = encoder.patch_embed(images)  # [B, N, C] where N = num_patches
    
    # Add CLS token
    cls_tokens = encoder.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, C]
    
    # Add positional embedding
    x = x + encoder.pos_embed
    x = encoder.pos_drop(x)
    
    # Pass through transformer blocks
    for blk in encoder.blocks:
        x = blk(x)
    
    # Normalize
    x = encoder.norm(x)
    
    # Extract patch tokens (skip CLS token at index 0)
    patch_tokens = x[:, 1:, :]  # [B, N, C]
    cls_token = x[:, 0, :]  # [B, C]
    return patch_tokens, cls_token

def extract_rad_dino_features(encoder, images):
    """
    Extract features from RadDino (HuggingFace DinoV2 based).
    
    先调用 encoder.preprocess() 利用 HF processor (do_rescale=False) 对 [0,1] tensor
    做 ImageNet normalize，再调用 encode()。等效于 forward() 但不带
    @torch.inference_mode()，保证训练时梯度可流通。
    
    Returns:
        patch_tokens: [B, H*W, C]
        cls_token: [B, C]
    """
    batch = encoder.preprocess(images).to(encoder.device)
    cls_token, patch_tokens = encoder.encode(batch)
    return patch_tokens, cls_token

def extract_chexzero_features(encoder, images):
    """
    Extract features from CheXzero (CLIP ViT, patch_size=32).
    
    Returns:
        patch_tokens: [B, H*W, C]
        cls_token: [B, C]
    """
    # CheXzero uses clip_vit.VisualTransformer
    # forward only returns CLS token [B, proj_dim], so we need manual extraction
    # Manual extraction: conv1 -> reshape -> permute -> add CLS -> transformer -> extract tokens
    x = encoder.conv1(images)  # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
    x = x.permute(0, 2, 1)  # [B, grid**2, width]
    
    # Add CLS token
    cls_embedding = encoder.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_embedding, x], dim=1)  # [B, grid**2 + 1, width]
    
    # Add positional embedding
    x = x + encoder.positional_embedding.to(x.dtype)
    x = encoder.ln_pre(x)
    
    # Transformer expects LND format
    x = x.permute(1, 0, 2)  # [grid**2 + 1, B, width]
    x = encoder.transformer(x)
    x = x.permute(1, 0, 2)  # [B, grid**2 + 1, width]
    
    # Extract CLS token and patch tokens
    cls_token = encoder.ln_post(x[:, 0, :])  # [B, width]
    if encoder.proj is not None:
        cls_token = cls_token @ encoder.proj
    
    # Patch tokens (before ln_post and proj)
    patch_tokens = x[:, 1:, :]  # [B, grid**2, width]
    
    return patch_tokens, cls_token

def extract_gloria_features(encoder, images):
    """
    Extract features from Gloria (DenseNet121 based).
    
    Returns:
        patch_tokens: [B, H*W, C]
        cls_token: [B, C]
    """
    # Gloria uses torchvision DenseNet121
    # We need the feature map before global pooling
    feat_map = encoder.features(images)  # [B, C, H, W]
    feat_map = torch.nn.functional.relu(feat_map, inplace=True)
    
    B, C, H, W = feat_map.shape
    patch_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, H*W, C]
    cls_token = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))  # [B, C, 1, 1]
    cls_token = torch.flatten(cls_token, 1)  # [B, C]
    
    return patch_tokens, cls_token

def extract_chess_features(encoder, images):
    """
    Extract features from CheSS (ResNet50 based).
    
    Returns:
        patch_tokens: [B, H*W, C]
        cls_token: [B, C]
    """
    # CheSS uses custom ResNet50 from resnet
    # Manual forward to get last feature map
    x = encoder.conv1(images)
    x = encoder.bn1(x)
    x = encoder.relu(x)
    x = encoder.maxpool(x)
    x = encoder.layer1(x)
    x = encoder.layer2(x)
    x = encoder.layer3(x)
    feat_map = encoder.layer4(x)  # [B, C, H, W]
    
    B, C, H, W = feat_map.shape
    patch_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, H*W, C]
    cls_token = encoder.avgpool(feat_map)  # [B, C, 1, 1]
    cls_token = torch.flatten(cls_token, 1)  # [B, C]
 
    return patch_tokens, cls_token

def get_extract_features_fn(model_name):
    model_name_lower = model_name.lower()
    
    if model_name_lower in ["ark_base", "ark_large", "our_ark_base", "our_ark_large"]:
        return extract_swin_features
    elif model_name_lower == "gloria":
        return extract_gloria_features
    elif model_name_lower == "chess":
        return extract_chess_features
    elif model_name_lower == "maco":
        return extract_vit_features
    elif model_name_lower == "chexzero":
        return extract_chexzero_features
    elif model_name_lower == "rad_dino":
        return extract_rad_dino_features
    else:
        return extract_vit_features


if __name__ == "__main__":
    import torch
    
    model_names = [
        "gloria",
        "chexzero",
        "maco",
        "rad_dino",
        "ark_base",
        "ark_large",
        "chess",
    ]
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_name}")
        print(f"{'='*50}")
        try:
            cfg = {
                "model_name": model_name,
            }
            encoder = create_vision_encoder(cfg)
            print(f"Successfully created {model_name} encoder")
            images = torch.randn(3, 3, 224, 224)
            if model_name == "ark_large":
                images = torch.randn(3, 3, 768, 768)
            elif model_name == "chess":
                images = torch.randn(3, 1, 512, 512)
            extract_features_fn = get_extract_features_fn(model_name)
            patch_tokens, cls_token = extract_features_fn(encoder, images)
            print(f"patch_tokens: {patch_tokens.shape}")
            print(f"cls_token: {cls_token.shape}")
        except Exception as e:
            print(f"Error creating {model_name} encoder: {e}")