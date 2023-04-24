# Copyright 2021 TUNiB inc .
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
# Contributed by abhilash1910

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoxLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceLinear


class GPTNeoxPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.hidden_size": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.num_attention_heads": config.num_attention_heads // world_size,
            
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(weight="attention.query_key_value.weight")
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.dense.weight",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="mlp.dense_h_to_4h.weight",
                bias="mlp.dense_h_to_4h.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="mlp.dense_4h_to_h.weight",
                bias="mlp.dense_4h_to_h.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return GPTNeoxLayer
