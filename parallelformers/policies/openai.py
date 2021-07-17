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

from transformers.models.openai.modeling_openai import Block

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceConv1D


class OpenAIGPTPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attn.split_size": config.n_embd // world_size,
            # 2. reduce number of heads
            "attn.n_head": config.n_head // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attn.c_attn.weight",
                bias="attn.c_attn.bias",
                n_fused=3,
                reversed=True,
            )
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attn.c_proj.weight",
                bias="attn.c_proj.bias",
                replace=AllReduceConv1D,
                reversed=True,
            )
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="mlp.c_fc.weight",
                bias="mlp.c_fc.bias",
                reversed=True,
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="mlp.c_proj.weight",
                bias="mlp.c_proj.bias",
                replace=AllReduceConv1D,
                reversed=True,
            )
        ]

    @staticmethod
    def original_layer_class():
        return Block
