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

from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from parallelformers.policies.base import Policy
from parallelformers.policies.base.policy import Layer
from parallelformers.utils import AllReduceConv1D


class GPT2Policy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attn.embed_dim": config.hidden_size // world_size,
            "attn.split_size": config.hidden_size // world_size,
            "crossattention.embed_dim": config.hidden_size // world_size,
            "crossattention.split_size": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attn.num_heads": config.num_attention_heads // world_size,
            "crossattention.num_heads": config.num_attention_heads // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attn.c_attn.weight",
                bias="attn.c_attn.bias",
                n_fused=3,
                reversed=True,
            ),
            Layer(
                weight="crossattention.c_attn.weight",
                bias="crossattention.c_attn.bias",
                n_fused=2,
                reversed=True,
                ignore_checker=True,
            ),
            Layer(
                weight="crossattention.q_attn.weight",
                bias="crossattention.q_attn.bias",
                reversed=True,
                ignore_checker=True,
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attn.c_proj.weight",
                bias="attn.c_proj.bias",
                replace=AllReduceConv1D,
                reversed=True,
            ),
            Layer(
                weight="crossattention.c_proj.weight",
                bias="crossattention.c_proj.bias",
                replace=AllReduceConv1D,
                reversed=True,
                ignore_checker=True,
            ),
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
        return GPT2Block
