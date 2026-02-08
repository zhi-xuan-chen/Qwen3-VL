#!/usr/bin/env python3
"""
MIMIC-CXR 测试集推理脚本

支持两种模型：
1. 原版 Qwen (sft_mimic_4b)：标准 LoRA 微调
2. 自定义视觉编码器 Qwen (sft_mimic_4b_custom_vision)：rad_dino / ark_base 等

用法示例：
    # 原版 Qwen
    python infer_mimic.py \
        --checkpoint_path /path/to/checkpoint-xxx \
        --test_json /path/to/mimic_qwen_test.json \
        --data_path /path/to/imgs-1024 \
        --output_file results/mimic_predictions.jsonl

    # 自定义视觉编码器
    python infer_mimic.py \
        --checkpoint_path /path/to/checkpoint-xxx \
        --test_json /path/to/mimic_qwen_test.json \
        --data_path /path/to/imgs-1024 \
        --output_file results/mimic_predictions.jsonl \
        --custom_vision_encoder ark_base
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# MIMIC 训练时的固定 prompt
DEFAULT_PROMPT = "Based on the chest X-ray image(s) provided, please generate the radiology report findings section."


def _make_abs_path(base: Path, p: str) -> str:
    path = str((base / p).resolve())
    if not path.startswith(("http://", "https://", "file://")):
        path = "file://" + path
    return path


def build_messages_for_inference(item: dict, base_path: Path) -> list:
    """构建推理用 messages（仅 user 轮，含 image + prompt）。"""
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    if not images:
        raise ValueError("No images in item")

    content = []
    for img in images:
        content.append({"type": "image", "image": _make_abs_path(base_path, img)})
    # 取 human 的 prompt 部分（去掉 <image> 占位后的文本）
    for turn in item["conversations"]:
        if turn["from"] == "human":
            text = turn["value"].replace("<image>", "").strip()
            if not text:
                text = DEFAULT_PROMPT
            content.append({"type": "text", "text": text})
            break
    else:
        content.append({"type": "text", "text": DEFAULT_PROMPT})

    return [{"role": "user", "content": content}]


def preprocess_standard(item: dict, base_path: Path, processor) -> dict:
    """标准 Qwen：用 processor 处理。"""
    messages = build_messages_for_inference(item, base_path)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def preprocess_custom_vision(item: dict, base_path: Path, processor, custom_vision_config: dict) -> dict:
    """自定义视觉编码器：手动构建 input_ids、pixel_values、image_grid_thw、position_ids。"""
    import re
    from qwenvl.data.rope2d import get_rope_index_3

    conversations = item["conversations"]
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    image_paths = [str((base_path / img).resolve()) for img in images]

    num_tokens = custom_vision_config["num_tokens"]
    transform = custom_vision_config["transform"]
    merge_size = custom_vision_config.get("merge_size", 2)

    # vision block
    vision_block = (
        "<|vision_start|>"
        + "<|image_pad|>" * num_tokens
        + "<|vision_end|>"
    )

    # 取 human 的 value，将 <image> 替换为 vision_block
    for turn in conversations:
        if turn["from"] == "human":
            text = turn["value"]
            break
    else:
        text = f"<image>\n{DEFAULT_PROMPT}"
    parts = re.split(r"(<image>)", text)
    user_text = ""
    for part in parts:
        user_text += vision_block if part == "<image>" else part

    tokenizer = processor.tokenizer
    messages = [{"role": "user", "content": user_text}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = torch.stack([transform(img) for img in pil_images])

    s = int(math.isqrt(num_tokens))
    if s * s != num_tokens:
        raise ValueError(f"num_tokens={num_tokens} 需为完全平方数")
    llm_h = llm_w = s
    grid_h = llm_h * merge_size
    grid_w = llm_w * merge_size
    image_grid_thw = torch.tensor(
        [[1, grid_h, grid_w]] * len(image_paths), dtype=torch.long
    )

    attention_mask = torch.ones_like(input_ids)
    position_ids, _ = get_rope_index_3(
        spatial_merge_size=merge_size,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "position_ids": position_ids,
    }


def load_model_standard(checkpoint_path: str, base_model: str | None, device: str = "cuda") -> tuple:
    """加载原版 Qwen + LoRA。"""
    from peft import PeftModel

    ckpt = Path(checkpoint_path)
    adapter_config_path = ckpt / "adapter_config.json"
    if base_model is None and adapter_config_path.exists():
        with open(adapter_config_path) as f:
            cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path")
    if not base_model:
        base_model = "Qwen/Qwen3-VL-4B-Instruct"

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, str(ckpt))
    model.eval()
    return model, processor


def load_model_custom_vision(
    checkpoint_path: str,
    base_model: str | None,
    custom_vision_encoder: str,
    custom_vision_pretrained_path: str | None,
    device: str = "cuda",
) -> tuple:
    """加载自定义视觉编码器 Qwen + LoRA。"""
    import types
    from peft import PeftModel
    from qwenvl.vision import build_custom_vision_for_qwen
    from qwenvl.vision.transforms import get_custom_image_transform
    from qwenvl.vision.utils import VIS_ENCODER_CONFIG

    ckpt = Path(checkpoint_path)
    adapter_config_path = ckpt / "adapter_config.json"
    if base_model is None and adapter_config_path.exists():
        with open(adapter_config_path) as f:
            cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path")
    if not base_model:
        base_model = "Qwen/Qwen3-VL-4B-Instruct"

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    llm_hidden_size = getattr(
        model.config, "hidden_size",
        getattr(model.config.text_config, "hidden_size", 3584),
    )
    encoder_cfg = {"model_name": custom_vision_encoder}
    if custom_vision_pretrained_path:
        encoder_cfg["pretrained_path"] = custom_vision_pretrained_path

    merge_size = getattr(
        processor.image_processor, "merge_size", 2
    )
    custom_visual = build_custom_vision_for_qwen(
        encoder_cfg=encoder_cfg,
        llm_hidden_size=llm_hidden_size,
        spatial_merge_size=merge_size,
    )
    custom_visual = custom_visual.to(dtype=torch.bfloat16, device=model.device)
    model.model.visual = custom_visual
    if hasattr(model.model.visual, "deepstack_visual_indexes"):
        model.model.visual.deepstack_visual_indexes = []

    _orig_model = model.model

    def _custom_get_image_features(self, pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(self.visual.dtype)
        result = self.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(result, (tuple, list)):
            image_embeds = result[0]
            deepstack_image_embeds = result[1] if len(result) > 1 else []
        else:
            image_embeds = result
            deepstack_image_embeds = []
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    _orig_model.get_image_features = types.MethodType(_custom_get_image_features, _orig_model)

    model = PeftModel.from_pretrained(model, str(ckpt))
    model.eval()

    cfg = VIS_ENCODER_CONFIG[custom_vision_encoder]
    custom_vision_config = {
        "num_tokens": cfg["num_tokens"],
        "transform": get_custom_image_transform(custom_vision_encoder),
        "merge_size": merge_size,
    }

    return model, processor, custom_vision_config


def get_ground_truth(item: dict) -> str:
    """从 conversations 中提取 gpt 的 findings。"""
    for turn in item["conversations"]:
        if turn["from"] == "gpt":
            return turn.get("value", "").strip()
    return ""


def run_inference(args):
    base_path = Path(args.data_path)
    test_json = Path(args.test_json)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_json) as f:
        test_data = json.load(f)

    use_custom_vision = bool(args.custom_vision_encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_custom_vision:
        model, processor, custom_vision_config = load_model_custom_vision(
            args.checkpoint_path,
            args.base_model,
            args.custom_vision_encoder,
            args.custom_vision_pretrained_path,
            device,
        )
        print(f"Loaded custom vision model: {args.custom_vision_encoder}")
    else:
        model, processor = load_model_standard(args.checkpoint_path, args.base_model, device)
        custom_vision_config = None
        print("Loaded standard Qwen model")

    results = []
    for idx, item in enumerate(tqdm(test_data, desc="Inference")):
        try:
            if use_custom_vision:
                inputs = preprocess_custom_vision(item, base_path, processor, custom_vision_config)
            else:
                inputs = preprocess_standard(item, base_path, processor)

            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    top_p=args.top_p,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = generated[:, input_len:]
            pred_text = processor.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            image_field = item.get("image")
            if isinstance(image_field, list):
                image_paths = image_field
            else:
                image_paths = [image_field]

            sample_id = item.get("id", idx)
            results.append({
                "id": sample_id,
                "image_paths": image_paths,
                "pred": pred_text,
                "gt": get_ground_truth(item),
            })
        except Exception as e:
            print(f"[WARN] Sample {idx} failed: {e}")
            sample_id = item.get("id", idx)
            results.append({
                "id": sample_id,
                "image_paths": item.get("image", []),
                "pred": "",
                "gt": get_ground_truth(item),
                "error": str(e),
            })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="MIMIC-CXR test set inference")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="微调后的 checkpoint 路径（含 adapter）",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/jhcnas4/XR/MIMIC-CXR/labels/qwen/mimic_qwen_test.json",
        help="MIMIC 测试集 JSON 路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/jhcnas4/XR/MIMIC-CXR/imgs-1024",
        help="图像根目录（与 JSON 中的相对路径拼接）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/mimic_predictions.jsonl",
        help="预测结果输出路径（JSONL）",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base 模型路径，不填则从 adapter_config.json 读取",
    )
    parser.add_argument(
        "--custom_vision_encoder",
        type=str,
        default=None,
        help="自定义视觉编码器名称，如 ark_base / rad_dino（不填则使用原版 Qwen）",
    )
    parser.add_argument(
        "--custom_vision_pretrained_path",
        type=str,
        default=None,
        help="our_ 开头 encoder 的预训练权重路径",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成长度",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度，0 表示贪婪解码",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top-p 采样",
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
