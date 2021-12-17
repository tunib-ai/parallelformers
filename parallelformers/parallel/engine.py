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
from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from parallelformers.parallel.replacing import TensorReplacer
from parallelformers.policies.base import Policy
from parallelformers.utils import rsetattr


class ParallelEngine(object):
    r"""
    Model parallelization processing engine

    Args:
        num_gpus (int): number of gpus
        backend (str): distributed backend (default=nccl)
        custom_policies (Union[Policy, List[Policy]]): user customized policy objects

    Notes:
        The parallelization process is performed through the following process.

        1) slice parallelizable tensors and replace original tensors on CPU
        2) upload parallelizable (replaced) tensors to multiple GPUs simultaneously
        3) upload non-parallelizable tensors to multiple GPUs (e.g. embedding, lm_head, ...)
    """

    def __init__(
        self,
        num_gpus: int,
        backend: str,
        custom_policies: Union[Policy, List[Policy]],
    ) -> None:

        self.num_gpus = num_gpus
        self.custom_policies = custom_policies
        self.mp_group = self.create_process_group(backend)
        # Create process group for model parallelization.

    def parallelize(self, model: nn.Module, fp16: bool) -> nn.Module:
        """
        Parallelize model to multiple GPUs

        Args:
            model (nn.Module): Huggingface pre-trained transformer model.
            fp16: (bool): whether use FP16 or not.

        Returns:
            nn.Module: parallelized model
        """

        super().__init__()
        replacer = TensorReplacer(
            model=model,
            fp16=fp16,
            mp_group=self.mp_group,
            num_gpus=self.num_gpus,
            custom_policies=self.custom_policies,
        )

        # Replace original layer to tensor sliced (megatron) layer.
        replacer.replace_modules()

        # Lazy GPU memory allocation (Only cpu tensors are loaded onto all gpus)
        # It's more memory-efficient than original implementation of DeepSpeed.
        for k, v in dict(model.state_dict()).items():
            if not v.is_cuda:
                if torch.is_tensor(v):
                    rsetattr(
                        model,
                        k + ".data",
                        v.to(torch.cuda.current_device()),
                    )

        return model

    def create_process_group(self, backend: str):
        """
        Create Pytorch distributed process group

        Args:
            backend (str): distributed backend

        Returns:
            ProcessGroupNCCL: process group for parallization
        """
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        new_group = dist.new_group([i for i in range(self.num_gpus)])

        return new_group
