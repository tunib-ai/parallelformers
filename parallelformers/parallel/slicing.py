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

import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist


class TensorSlicer(object):
    r"""
    An object that slices tensors into rows or columns as described in the Megatron LM paper

    Args:
        mp_group (torch.distributed.ProcessGroupNCCL): Distributed group for model parallelism
    """

    def __init__(self, mp_group) -> None:
        if dist.is_initialized() and mp_group is not None:
            self.gpu_index = dist.get_rank(group=mp_group)
            self.world_size = int(os.getenv("WORLD_SIZE"))
        else:
            self.gpu_index = 0
            self.world_size = 1

    def slice_tensor(
        self,
        tensor: Dict,
        attributes: Dict,
        dim: int,
        is_bias: bool,
    ) -> Tuple:
        """
        Slice tensors into rows or columns as described in the Megatron LM paper

        Args:
            tensor (Dict): tensor dictionaries
            attributes (Dict): attributes dictionaries
            dim (int): dimension for slicing
            is_bias (bool): whether tensor is bias or not

        Returns:
            Tuple: tuple of sliced tensors
        """
        if not tensor:
            return (None,)

        n_fused_list, reversed_list = [], []
        for (k_tensor, _), (k_attr, v_attr) in zip(
            tensor.items(),
            attributes.items(),
        ):
            if k_tensor == k_attr:
                n_fused, reversed = v_attr
                n_fused_list.append(n_fused)
                reversed_list.append(reversed)

        tensor = list(tensor.values())
        slices = []

        for proj_layer, n_fused, reversed in zip(
            tensor,
            n_fused_list,
            reversed_list,
        ):
            device = torch.cuda.current_device()
            dim = dim if not reversed or is_bias else abs(dim - 1)
            n_fused = 1 if not n_fused else n_fused

            proj_layer = proj_layer.chunk(
                n_fused * self.world_size,
                dim=dim,
            )

            if n_fused > 1:
                ranks = (len(proj_layer) + self.world_size - 1) // self.world_size
                proj_layer = [
                    proj_layer[i * self.world_size : (i + 1) * self.world_size]
                    for i in range(ranks)
                ]
                proj_layer = list(
                    map(lambda x: torch.cat([*x], dim=-1), zip(*proj_layer))
                )

            proj_layer = proj_layer[self.gpu_index].to(device)
            slices.append(proj_layer)

        return tuple(slices)

    def slice_weight_and_bias(
        self,
        policy_inputs: Tuple,
        attributes: Tuple,
        dim: int,
        slice_bias: bool,
    ) -> Tuple:
        """
        Slice weight and bias for model parallelization

        Args:
            policy_inputs (Tuple): tuple of weight and bias dictionaries
            attributes (Tuple): tuple of weight attributes and bias attributes dictionaries
            dim (int): dimension for slicing
            slice_bias (bool): whether slice bias or not

        Returns:
            Tuple: tuple of weights and biases
        """
        weight, bias = policy_inputs
        w_attr, b_attr = attributes
        weight = self.slice_tensor(
            weight,
            w_attr,
            dim,
            is_bias=False,
        )

        if slice_bias:
            bias = self.slice_tensor(
                bias,
                b_attr,
                0,
                is_bias=True,
            )
        else:
            bias = tuple(bias.values())

        return weight, bias

    def column_slice(
        self,
        policy_inputs: Tuple,
        attributes: Tuple,
    ) -> Tuple:
        """
        Slice tensors in the column direction.

        Args:
            policy_inputs (Tuple): tuple of weight and bias dictionaries
            attributes (Tuple): tuple of weight attributes and bias attributes dictionaries

        Returns:
            Tuple: tuple of weights and biases
        """
        return self.slice_weight_and_bias(
            policy_inputs,
            attributes=attributes,
            dim=0,
            slice_bias=True,
        )

    def row_slice(
        self,
        policy_inputs: Tuple,
        attributes: Tuple,
    ) -> Tuple:
        """
        Slice tensors in the row direction.

        Args:
            policy_inputs (Tuple): tuple of weight and bias dictionaries
            attributes (Tuple): tuple of weight attributes and bias attributes dictionaries

        Returns:
            Tuple: tuple of weights and biases
        """
        return self.slice_weight_and_bias(
            policy_inputs,
            attributes=attributes,
            dim=1,
            slice_bias=False,
        )
