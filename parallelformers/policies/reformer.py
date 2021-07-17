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

from transformers.models.reformer.modeling_reformer import ReformerLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils.dist_utils import AllReduceLinear


class ReformerPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "self_attn.all_head_size": config.hidden_size // world_size,
            "self_attn.hidden_size": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.self_attention.num_attention_heads": config.num_attention_heads
            // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.self_attention.query_key.weight",
                ignore_checker=True,
            ),
            Layer(
                weight="attention.self_attention.query.weight",
                ignore_checker=True,
            ),
            Layer(
                weight="attention.self_attention.key.weight",
                ignore_checker=True,
            ),
            Layer(weight="attention.self_attention.value.weight"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.output.dense.weight",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="feed_forward.dense.dense.weight",
                bias="feed_forward.dense.dense.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="feed_forward.output.dense.weight",
                bias="feed_forward.output.dense.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return ReformerLayer
