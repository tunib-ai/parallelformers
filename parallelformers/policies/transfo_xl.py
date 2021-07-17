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

from transformers.models.transfo_xl.modeling_transfo_xl import (
    RelPartialLearnableDecoderLayer,
)

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils.dist_utils import AllReduceLinear


class TransfoXLPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "dec_attn.n_head": config.n_head // world_size,
            # 2. reduce number of heads
            "dec_attn.d_model": config.d_model // world_size,
        }

    @staticmethod
    def attn_qkv():
        # TransfoXL's qkv net is reversed.
        return [
            Layer(
                weight="dec_attn.qkv_net.weight",
                n_fused=3,
                reversed=True,
            ),
            Layer(weight="dec_attn.r_net.weight"),
            Layer(bias="dec_attn.r_r_bias"),
            Layer(bias="dec_attn.r_w_bias"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="dec_attn.o_net.weight",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="pos_ff.CoreNet[0].weight",
                bias="pos_ff.CoreNet[0].bias",
            )
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="pos_ff.CoreNet[3].weight",
                bias="pos_ff.CoreNet[3].bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return RelPartialLearnableDecoderLayer
