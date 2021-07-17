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

from transformers.models.ibert.modeling_ibert import IBertLayer

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils.dist_utils import AllReduceQuantLinear


class IBertPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attention.self.all_head_size": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attention.self.num_attention_heads": config.num_attention_heads
            // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="attention.self.query.weight",
                bias="attention.self.query.bias",
            ),
            Layer(
                weight="attention.self.key.weight",
                bias="attention.self.key.bias",
            ),
            Layer(
                weight="attention.self.value.weight",
                bias="attention.self.value.bias",
            ),
            Layer(
                weight="attention.self.query.weight_integer",
                bias="attention.self.query.bias_integer",
            ),
            Layer(
                weight="attention.self.key.weight_integer",
                bias="attention.self.key.bias_integer",
            ),
            Layer(
                weight="attention.self.value.weight_integer",
                bias="attention.self.value.bias_integer",
            ),
            Layer(bias="attention.self.query.fc_scaling_factor"),
            Layer(bias="attention.self.key.fc_scaling_factor"),
            Layer(bias="attention.self.value.fc_scaling_factor"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attention.output.dense.weight",
                bias="attention.output.dense.bias",
                replace=AllReduceQuantLinear,
            ),
            Layer(
                weight="attention.output.dense.weight_integer",
                bias="attention.output.dense.bias_integer",
            ),
            Layer(bias="attention.output.dense.fc_scaling_factor"),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="intermediate.dense.weight",
                bias="intermediate.dense.bias",
            ),
            Layer(
                weight="intermediate.dense.weight_integer",
                bias="intermediate.dense.bias_integer",
            ),
            Layer(bias="intermediate.dense.fc_scaling_factor"),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="output.dense.weight",
                bias="output.dense.bias",
                replace=AllReduceQuantLinear,
            ),
            Layer(
                weight="output.dense.weight_integer",
                bias="output.dense.bias_integer",
            ),
            Layer(bias="output.dense.fc_scaling_factor"),
        ]

    @staticmethod
    def original_layer_class():
        return IBertLayer
