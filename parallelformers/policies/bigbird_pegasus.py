from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
    BigBirdPegasusDecoderLayer,
    BigBirdPegasusEncoderLayer,
)

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_bart import BartAttention_
from parallelformers.utils.dist_utils import AllReduceLinear


class BigBirdPegasusEncoderPolicy(Policy):
    @staticmethod
    def replace_modules():
        return {
            "BigBirdPegasusDecoderAttention": BartAttention_,
        }

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
        return BigBirdPegasusEncoderLayer


class BigBirdPegasusDecoderPolicy(Policy):
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
            "BigBirdPegasusDecoderAttention": BartAttention_,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(weight="self_attn.q_proj.weight"),
            Layer(weight="self_attn.k_proj.weight"),
            Layer(weight="self_attn.v_proj.weight"),
            Layer(weight="encoder_attn.q_proj.weight"),
            Layer(weight="encoder_attn.k_proj.weight"),
            Layer(weight="encoder_attn.v_proj.weight"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="self_attn.out_proj.weight",
                replace=AllReduceLinear,
            ),
            Layer(
                weight="encoder_attn.out_proj.weight",
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
        return BigBirdPegasusDecoderLayer
