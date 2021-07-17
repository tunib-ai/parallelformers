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

from transformers.models.xlm.modeling_xlm import (
    MultiHeadAttention,
    TransformerFFN,
)

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils.dist_utils import AllReduceLinear


class XLMAttentionPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "dim": config.emb_dim // world_size,
            # 2. reduce number of heads
            "n_heads": config.n_heads // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="q_lin.weight",
                bias="q_lin.bias",
            ),
            Layer(
                weight="k_lin.weight",
                bias="k_lin.bias",
            ),
            Layer(
                weight="v_lin.weight",
                bias="v_lin.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="out_lin.weight",
                bias="out_lin.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return MultiHeadAttention


class XLMMLPPolicy(Policy):
    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="lin1.weight",
                bias="lin1.bias",
            )
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="lin2.weight",
                bias="lin2.bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return TransformerFFN
