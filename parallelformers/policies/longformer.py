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

from transformers.models.longformer.modeling_longformer import LongformerLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_longformer import (
    LongformerSelfAttention_,
)
from parallelformers.utils.dist_utils import AllReduceLinear


class LongformerPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.self.embed_dim": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.self.num_heads": config.num_attention_heads // world_size,
        }

    @staticmethod
    def replace_modules():
        return {"LongformerSelfAttention": LongformerSelfAttention_}

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
                weight="attention.self.query_global.weight",
                bias="attention.self.query_global.bias",
            ),
            Layer(
                weight="attention.self.key_global.weight",
                bias="attention.self.key_global.bias",
            ),
            Layer(
                weight="attention.self.value_global.weight",
                bias="attention.self.value_global.bias",
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
        return LongformerLayer
