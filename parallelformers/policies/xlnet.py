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

from transformers.models.xlnet.modeling_xlnet import XLNetLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.transformers.modeling_xlnet import XLNetRelativeAttention_
from parallelformers.utils.dist_utils import AllReduceLinear


class XLNetPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "rel_attn.n_head": config.n_head // world_size,
            # 2. reduce number of heads
            "rel_attn.d_model": config.d_model // world_size,
        }

    @staticmethod
    def replace_modules():
        return {
            "XLNetRelativeAttention": XLNetRelativeAttention_,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(weight="rel_attn.q", reversed=True),
            Layer(weight="rel_attn.k"),
            Layer(weight="rel_attn.v"),
            Layer(weight="rel_attn.r"),
            Layer(bias="rel_attn.r_r_bias"),
            Layer(bias="rel_attn.r_s_bias"),
            Layer(bias="rel_attn.r_w_bias"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(weight="rel_attn.o"),
            Layer(weight="rel_attn.seg_embed"),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="ff.layer_1.weight",
                bias="ff.layer_1.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="ff.layer_2.weight",
                bias="ff.layer_2.bias",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class():
        return XLNetLayer
