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

from transformers.models.ctrl.modeling_ctrl import EncoderLayer

from parallelformers.policies.base import Policy
from parallelformers.policies.base.policy import Layer
from parallelformers.utils import AllReduceLinear


class CTRLPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "multi_head_attention.d_model_size": config.n_embd // world_size,
            # 2. reduce number of heads
            "multi_head_attention.num_heads": config.n_head // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="multi_head_attention.Wq.weight",
                bias="multi_head_attention.Wq.bias",
            ),
            Layer(
                weight="multi_head_attention.Wk.weight",
                bias="multi_head_attention.Wk.bias",
            ),
            Layer(
                weight="multi_head_attention.Wv.weight",
                bias="multi_head_attention.Wv.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="multi_head_attention.dense.weight",
                bias="multi_head_attention.dense.bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="ffn[0].weight",
                bias="ffn[0].bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="ffn[2].weight",
                bias="ffn[2].bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return EncoderLayer
