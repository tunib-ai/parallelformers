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

from transformers.models.distilbert.modeling_distilbert import TransformerBlock

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceLinear


class DistilBertPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.dim": config.dim // world_size,
            # 2. reduce number of heads
            "attention.n_heads": config.n_heads // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.q_lin.weight",
                bias="attention.q_lin.bias",
            ),
            Layer(
                weight="attention.k_lin.weight",
                bias="attention.k_lin.bias",
            ),
            Layer(
                weight="attention.v_lin.weight",
                bias="attention.v_lin.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.out_lin.weight",
                bias="attention.out_lin.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="ffn.lin1.weight",
                bias="ffn.lin1.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="ffn.lin2.weight",
                bias="ffn.lin2.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return TransformerBlock
