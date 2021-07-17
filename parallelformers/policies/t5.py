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

from transformers.models.t5.modeling_t5 import T5Block

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceLinear


class T5Policy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "layer[0].SelfAttention.d_model": config.hidden_size // world_size,
            "layer[1].EncDecAttention.d_model": config.hidden_size // world_size,
            # 2. reduce number of heads
            "layer[0].SelfAttention.n_heads": config.num_attention_heads // world_size,
            "layer[1].EncDecAttention.n_heads": config.num_attention_heads
            // world_size,
            # 3. reduce inner dim
            "layer[0].SelfAttention.inner_dim": (
                config.d_kv * config.num_attention_heads
            )
            // world_size,
            "layer[1].EncDecAttention.inner_dim": (
                config.d_kv * config.num_attention_heads
            )
            // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(weight="layer[0].SelfAttention.q.weight"),
            Layer(weight="layer[0].SelfAttention.k.weight"),
            Layer(weight="layer[0].SelfAttention.v.weight"),
            Layer(
                weight="layer[1].EncDecAttention.q.weight",
                ignore_checker=True,
            ),
            Layer(
                weight="layer[1].EncDecAttention.k.weight",
                ignore_checker=True,
            ),
            Layer(
                weight="layer[1].EncDecAttention.v.weight",
                ignore_checker=True,
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="layer[0].SelfAttention.o.weight",
                replace=AllReduceLinear,
            ),
            Layer(
                weight="layer[1].EncDecAttention.o.weight",
                replace=AllReduceLinear,
                ignore_checker=True,
            ),
            Layer(
                weight="layer[0].SelfAttention.relative_attention_bias.weight",
                ignore_checker=True,
            ),
            Layer(
                weight="layer[1].EncDecAttention.relative_attention_bias.weight",
                ignore_checker=True,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            # relu
            Layer(
                weight="layer[-1].DenseReluDense.wi.weight",
                ignore_checker=True,
            ),
            # gated gelu
            Layer(
                weight="layer[-1].DenseReluDense.wi_0.weight",
                ignore_checker=True,
            ),
            Layer(
                weight="layer[-1].DenseReluDense.wi_1.weight",
                ignore_checker=True,
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="layer[-1].DenseReluDense.wo.weight",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return T5Block
