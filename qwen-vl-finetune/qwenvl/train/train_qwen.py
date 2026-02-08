# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    # 可选：替换为自定义视觉 encoder
    if model_args.custom_vision_encoder:
        import types
        from qwenvl.vision import build_custom_vision_for_qwen

        llm_hidden_size = getattr(
            model.config, "hidden_size",
            getattr(model.config.text_config, "hidden_size", 3584),
        )
        encoder_cfg = {"model_name": model_args.custom_vision_encoder}
        if model_args.custom_vision_pretrained_path:
            encoder_cfg["pretrained_path"] = model_args.custom_vision_pretrained_path

        # 获取 processor 的 merge_size，使 wrapper.spatial_merge_size 与数据管道一致
        _merge_size = getattr(
            AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor,
            "merge_size", 2,
        )

        custom_visual = build_custom_vision_for_qwen(
            encoder_cfg=encoder_cfg,
            llm_hidden_size=llm_hidden_size,
            spatial_merge_size=_merge_size,
        )
        custom_visual = custom_visual.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            device=model.device,
        )
        model.model.visual = custom_visual
        # 关闭 DeepStack（自定义 encoder 无多层 ViT 输出）
        if hasattr(model.model, "visual") and hasattr(model.model.visual, "deepstack_visual_indexes"):
            model.model.visual.deepstack_visual_indexes = []

        # Monkey-patch get_image_features：避免在 DeepSpeed ZeRO-3 下
        # tuple unpacking 与自定义 visual 返回值不兼容的问题
        _orig_model = model.model  # Qwen3VLModel 实例

        def _custom_get_image_features(self, pixel_values, image_grid_thw=None):
            pixel_values = pixel_values.type(self.visual.dtype)
            result = self.visual(pixel_values, grid_thw=image_grid_thw)
            # 兼容 tuple / list / 其他可迭代返回
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

        rank0_print(f"Replaced model.visual with custom encoder: {model_args.custom_vision_encoder}")
        rank0_print(f"  spatial_merge_size={_merge_size}")
        # 同步到 data_args，使数据管道使用自定义 transform 而非 Qwen image processor
        data_args.custom_vision_encoder = model_args.custom_vision_encoder

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # LoRA 模式下，tune_mm_mlp 仍需生效：解冻 merger，医学影像等域适应需要投影层适配
        if model_args.tune_mm_mlp:
            base = getattr(model, "base_model", model) or model
            inner = getattr(base, "model", base)
            visual = getattr(inner, "visual", None) or getattr(base, "visual", None) or getattr(model, "visual", None)
            if visual is not None and hasattr(visual, "merger"):
                for p in visual.merger.parameters():
                    p.requires_grad = True
                rank0_print("LoRA mode: merger (tune_mm_mlp) set to trainable")

        if torch.distributed.get_rank() == 0:
            model.print_trainable_parameters()
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            visual = getattr(model, "visual", model.model.visual)
            if hasattr(visual, "print_trainable_parameters"):
                visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
