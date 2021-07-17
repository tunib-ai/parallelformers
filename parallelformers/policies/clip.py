from transformers.models.clip.modeling_clip import (
    CLIPEncoderLayer,
    CLIPTextTransformer,
    CLIPVisionTransformer,
)

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_clip import CLIPAttention_
from parallelformers.utils import AllReduceLinear


class CLIPTextPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        arguments = {
            f"encoder.layers[{i}].self_attn.embed_dim": config.text_config.hidden_size
            // world_size
            for i in range(config.text_config.num_hidden_layers)
        }

        arguments.update(
            {
                f"encoder.layers[{i}].self_attn.num_heads": config.text_config.num_attention_heads
                // world_size
                for i in range(config.text_config.num_hidden_layers)
            }
        )

        return arguments

    @staticmethod
    def original_layer_class():
        return CLIPTextTransformer


class CLIPVisionPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        arguments = {
            f"encoder.layers[{i}].self_attn.embed_dim": config.vision_config.hidden_size
            // world_size
            for i in range(config.vision_config.num_hidden_layers)
        }

        arguments.update(
            {
                f"encoder.layers[{i}].self_attn.num_heads": config.vision_config.num_attention_heads
                // world_size
                for i in range(config.vision_config.num_hidden_layers)
            }
        )

        return arguments

    @staticmethod
    def original_layer_class():
        return CLIPVisionTransformer


class CLIPLayerPolicy(Policy):
    @staticmethod
    def replace_modules():
        return {
            "CLIPAttention": CLIPAttention_,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="self_attn.q_proj.weight",
                bias="self_attn.q_proj.bias",
            ),
            Layer(
                weight="self_attn.k_proj.weight",
                bias="self_attn.k_proj.bias",
            ),
            Layer(
                weight="self_attn.v_proj.weight",
                bias="self_attn.v_proj.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="self_attn.out_proj.weight",
                bias="self_attn.out_proj.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="mlp.fc1.weight",
                bias="mlp.fc1.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="mlp.fc2.weight",
                bias="mlp.fc2.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return CLIPEncoderLayer
