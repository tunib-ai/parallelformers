from transformers.models.detr.modeling_detr import (
    DetrDecoderLayer,
    DetrEncoderLayer,
)

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_detr import DetrAttention_
from parallelformers.utils import AllReduceLinear


class DetrEncoderPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "self_attn.embed_dim": config.hidden_size // world_size,
            # 2. reduce number of heads
            "self_attn.num_heads": config.encoder_attention_heads // world_size,
        }

    @staticmethod
    def replace_modules():
        return {
            "DetrAttention": DetrAttention_,
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
                weight="fc1.weight",
                bias="fc1.bias",
            )
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="fc2.weight",
                bias="fc2.bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return DetrEncoderLayer


class DetrDecoderPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "self_attn.embed_dim": config.d_model // world_size,
            "encoder_attn.embed_dim": config.d_model // world_size,
            # 2. reduce number of heads
            "self_attn.num_heads": config.decoder_attention_heads // world_size,
            "encoder_attn.num_heads": config.decoder_attention_heads // world_size,
        }

    @staticmethod
    def replace_modules():
        return {
            "DetrAttention": DetrAttention_,
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
            Layer(
                weight="encoder_attn.q_proj.weight",
                bias="encoder_attn.q_proj.bias",
            ),
            Layer(
                weight="encoder_attn.k_proj.weight",
                bias="encoder_attn.k_proj.bias",
            ),
            Layer(
                weight="encoder_attn.v_proj.weight",
                bias="encoder_attn.v_proj.bias",
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
            Layer(
                weight="encoder_attn.out_proj.weight",
                bias="encoder_attn.out_proj.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="fc1.weight",
                bias="fc1.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="fc2.weight",
                bias="fc2.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return DetrDecoderLayer
