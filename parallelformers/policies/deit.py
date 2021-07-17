from transformers.models.deit.modeling_deit import DeiTLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceLinear


class DeiTPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.attention.all_head_size": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.attention.num_attention_heads": config.num_attention_heads
            // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.attention.query.weight",
                bias="attention.attention.query.bias",
            ),
            Layer(
                weight="attention.attention.key.weight",
                bias="attention.attention.key.bias",
            ),
            Layer(
                weight="attention.attention.value.weight",
                bias="attention.attention.value.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.output.dense.weight",
                bias="attention.output.dense.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="intermediate.dense.weight",
                bias="intermediate.dense.bias",
            )
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="output.dense.weight",
                bias="output.dense.bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return DeiTLayer
