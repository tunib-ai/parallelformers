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

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import Linear
from transformers.modeling_utils import Conv1D
from transformers.models.ibert.quant_modules import (
    QuantLinear,
    symmetric_linear_quantization_params,
)


class ParallelModule(nn.Module):
    """Parents of all parallel layer classes"""

    def __init__(self):
        super().__init__()
        self.mp_group = None

    def allreduce(self, outputs):
        if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:

            dist.all_reduce(
                outputs,
                group=self.mp_group,
            )
        return outputs


class AllReduceLinear(Linear, ParallelModule):
    """All-reduce linear layer"""

    def forward(self, input: Tensor) -> Tensor:
        outputs = input.matmul(self.weight.t())

        self.allreduce(outputs)
        if self.bias is not None:
            outputs += self.bias

        return outputs


class AllReduceConv1D(Conv1D, ParallelModule):
    """All-reduce convolution 1D layer for GPT models"""

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        outputs = torch.mm(x.view(-1, x.size(-1)), self.weight)

        self.allreduce(outputs)
        if self.bias is not None:
            outputs += self.bias

        return outputs.view(*size_out)


class AllReduceQuantLinear(QuantLinear, ParallelModule):
    """All-reduce quantized linear layer for IBert models"""

    def allreduce_linear_layer(self, input, weight, bias=None):
        outputs = input.matmul(weight.t())

        self.allreduce(outputs)
        if bias is not None:
            outputs += bias

        return outputs

    def forward(self, x, prev_act_scaling_factor=None):
        if not self.quant_mode:
            return self.allreduce_linear_layer(x, self.weight, self.bias), None

        # assert that prev_act_scaling_factor is a scalar tensor
        assert prev_act_scaling_factor is not None and prev_act_scaling_factor.shape == (
            1,
        ), (
            "Input activation to the QuantLinear layer should be globally (non-channel-wise) quantized. "
            "Please add a QuantAct layer with `per_channel = True` before this QuantAct layer"
        )

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        self.fc_scaling_factor = symmetric_linear_quantization_params(
            self.weight_bit, w_min, w_max, self.per_channel
        )
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor
        )

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, False, bias_scaling_factor
            )

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return (
            self.allreduce_linear_layer(x_int, self.weight_integer, self.bias_integer)
            * bias_scaling_factor,
            bias_scaling_factor,
        )
