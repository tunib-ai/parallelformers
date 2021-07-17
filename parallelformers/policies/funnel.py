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

from transformers.models.funnel.modeling_funnel import FunnelLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_funnel import (
    FunnelRelMultiheadAttention_,
)
from parallelformers.utils.dist_utils import AllReduceLinear


class FunnelPolicy(Policy):
    @staticmethod
    def replace_modules():
        return {
            "FunnelRelMultiheadAttention": FunnelRelMultiheadAttention_,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.q_head.weight",
            ),
            Layer(
                weight="attention.k_head.weight",
                bias="attention.k_head.bias",
            ),
            Layer(
                weight="attention.v_head.weight",
                bias="attention.v_head.bias",
            ),
            Layer(bias="attention.r_r_bias"),
            Layer(bias="attention.r_s_bias"),
            Layer(bias="attention.r_w_bias"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.post_proj.weight",
                bias="attention.post_proj.bias",
                replace=AllReduceLinear,
            ),
            Layer(weight="attention.seg_embed"),
            Layer(weight="attention.r_kernel"),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="ffn.linear_1.weight",
                bias="ffn.linear_1.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="ffn.linear_2.weight",
                bias="ffn.linear_2.bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return FunnelLayer
