# Copyright 2021 TUNiB inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_bart import BartAttention_
from parallelformers.utils import AllReduceLinear


class Wav2VecPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.embed_dim": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.num_heads": config.num_attention_heads // world_size,
        }

    @staticmethod
    def replace_modules():
        return {
            "Wav2Vec2Attention": BartAttention_,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.q_proj.weight",
                bias="attention.q_proj.bias",
            ),
            Layer(
                weight="attention.k_proj.weight",
                bias="attention.k_proj.bias",
            ),
            Layer(
                weight="attention.v_proj.weight",
                bias="attention.v_proj.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.out_proj.weight",
                bias="attention.out_proj.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="feed_forward.intermediate_dense.weight",
                bias="feed_forward.intermediate_dense.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="feed_forward.output_dense.weight",
                bias="feed_forward.output_dense.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return Wav2Vec2EncoderLayer
