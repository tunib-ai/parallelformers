from transformers.models.roformer.modeling_roformer import RoFormerLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceLinear


class RoformerPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.self.all_head_size": config.hidden_size // world_size,
            "crossattention.self.all_head_size": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.self.num_attention_heads": config.num_attention_heads
            // world_size,
            "crossattention.self.num_attention_heads": config.num_attention_heads
            // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.self.query.weight",
                bias="attention.self.query.bias",
            ),
            Layer(
                weight="attention.self.key.weight",
                bias="attention.self.key.bias",
            ),
            Layer(
                weight="attention.self.value.weight",
                bias="attention.self.value.bias",
            ),
            Layer(
                weight="crossattention.self.query.weight",
                bias="crossattention.self.query.bias",
                ignore_checker=True,
            ),
            Layer(
                weight="crossattention.self.key.weight",
                bias="crossattention.self.key.bias",
                ignore_checker=True,
            ),
            Layer(
                weight="crossattention.self.value.weight",
                bias="crossattention.self.value.bias",
                ignore_checker=True,
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
            Layer(
                weight="crossattention.output.dense.weight",
                bias="crossattention.output.dense.bias",
                replace=AllReduceLinear,
                ignore_checker=True,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="intermediate.dense.weight",
                bias="intermediate.dense.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="output.dense.weight",
                bias="output.dense.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return RoFormerLayer
