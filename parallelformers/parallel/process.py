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

import copyreg
import io
import os
import pickle
import random
import traceback
import types
from contextlib import suppress
from dataclasses import _is_dataclass_instance, asdict
from inspect import signature
from time import time
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from transformers.file_utils import ModelOutput

from parallelformers.parallel.engine import ParallelEngine
from parallelformers.policies.base import Policy


class ForkingPickler(pickle.Pickler):
    """Copy of ForkingPickler of `multiprocessing` module"""

    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    def __init__(self, *args):
        """Constructor of ForkingPickler"""
        super().__init__(*args)
        self.dispatch_table = self._copyreg_dispatch_table.copy()
        self.dispatch_table.update(self._extra_reducers)

    @classmethod
    def register(cls, type, reduce) -> None:
        """Register reduce methods for multiprocessing"""
        cls._extra_reducers[type] = reduce

    @classmethod
    def dumps(cls, obj: Any, protocol=None) -> memoryview:
        """Dump objects for multiprocessing"""
        buf = io.BytesIO()
        cls(buf, protocol).dump(obj)
        return buf.getbuffer()

    loads = pickle.loads


class ParallelProcess(mp.Process):
    r"""
    Parallelization process class

    Args:
        model (nn.Module): model weights
        fp16: (bool): whether use FP16 or not.
        rank (int): current GPU rank
        num_gpus (int): number of gpus for parallelization
        inputs_queue (mp.Queue): input data queue from user
        outputs_queue (mp.Queue): output data queue from model
        parallel_mutex (mp.Event): mutex object to notify parallelization state
        inference_mutex (mp.Event): mutex object to notify inference state
        verbose (str): turn on gpu summary
        backend (str): distributed process backend
        custom_policies (Union[Policy, List[Policy]]): user customized policy objects

    Notes:
        ParallelProcess object handles below two tasks.

        1) Parallelize the model
        2) Handle the inference state
    """

    _memory_logger = {
        "memory_summary": torch.cuda.memory_summary,
        "memory_reserved": torch.cuda.memory_reserved,
        "memory_cached": torch.cuda.memory_reserved,
        "memory_allocated": torch.cuda.memory_allocated,
    }

    def __init__(
        self,
        model: nn.Module,
        fp16: bool,
        rank: int,
        num_gpus: int,
        inputs_queue: mp.Queue,
        outputs_queue: mp.Queue,
        parallel_mutex: mp.Event,
        inference_mutex: mp.Event,
        verbose: str,
        backend: str,
        custom_policies: Union[Policy, List[Policy]],
        seed: int,
    ) -> None:
        super().__init__()
        self.set_environ(rank)
        self.model = model
        self.fp16 = fp16
        self.num_gpus = num_gpus
        self.inputs_queue = inputs_queue
        self.outputs_queue = outputs_queue
        self.parallel_mutex = parallel_mutex
        self.inference_mutex = inference_mutex
        self.verbose = verbose
        self.backend = backend
        self.custom_policies = custom_policies
        self.seed = seed

    def set_environ(self, rank: int) -> None:
        """
        Set environment variable of current process

        Args:
            rank (int): current GPU rank
        """
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)

    def destroy(self) -> None:
        """Callback that executed when the process terminates."""
        for method in self._memory_logger:
            setattr(self.model, method, None)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def inference(self, model: nn.Module) -> None:
        """
        Handle inference state.
        If an inference request is occurred from main process,
        Infer the model and pass the output to main process.

        Args:
            model (nn.Module): model weight
        """
        if self.seed is None:
            seed = torch.tensor(int(time())).cuda()
            dist.broadcast(seed, src=0)
            seed = seed.item()
        else:
            seed = self.seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        with suppress():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        while True:
            try:
                self.inference_mutex.wait()
                self.inference_mutex.clear()
                device = torch.cuda.current_device()

                # consumer part
                inputs, kwargs, fn_name = self.inputs_queue.get()
                inputs_, kwargs_ = [], {}

                for i in inputs:
                    if torch.is_tensor(i):
                        i = i.clone().to(device)
                    inputs_.append(i)

                for k in kwargs:
                    if torch.is_tensor(kwargs[k]):
                        kwargs[k] = kwargs[k].clone().to(device)
                    kwargs_[k] = kwargs[k]

                if fn_name not in self._memory_logger:
                    function_ = getattr(model, fn_name)
                    n_params = len(signature(function_).parameters)

                    if n_params > 0:
                        outputs = function_(
                            *inputs_,
                            **kwargs_,
                        )
                    else:
                        outputs = function_()
                else:
                    outputs = (
                        f"cuda:{device}",
                        str(self._memory_logger[fn_name](device)),
                    )

                if fn_name in ["cuda", "cpu", "to"]:
                    break

                # check picklable
                outputs = self.check_picklable(outputs)

                if isinstance(outputs, types.GeneratorType):
                    outputs = list(outputs)

                # producer part
                self.outputs_queue.put(outputs)

            except BaseException:
                traceback.print_exc()
                break

    def check_picklable(self, obj: Any) -> Any:
        """
        Check object is picklable.
        If it is not picklable, this method will change the dataclass instance to a dictionary.
        It is is not dataclass raise exception.

        Args:
            obj (Any): object to check picklable

        Returns:
            Any: picklable object
        """
        try:
            pickle.loads(ForkingPickler.dumps(obj).tobytes())
        except BaseException:
            if _is_dataclass_instance(obj) or isinstance(obj, ModelOutput):
                _obj = asdict(obj)
                _obj["orig_dataclass_type"] = obj.__class__
                obj = _obj
            else:
                raise Exception(
                    f"Type '{obj.__class__}' can't be pickled. "
                    f"Please check type of model output !"
                )

        return obj

    @torch.no_grad()
    def run(self) -> None:
        """Start parallelization process"""
        engine = ParallelEngine(
            num_gpus=self.num_gpus,
            backend=self.backend,
            custom_policies=self.custom_policies,
        )

        try:
            self.model = engine.parallelize(self.model, self.fp16)
            self.parallel_mutex.set()

            if self.verbose:
                if self.verbose is True or self.verbose.lower() == "simple":
                    device = torch.cuda.current_device()
                    print(f"GPU {device} alloc: {torch.cuda.memory_allocated(device)}")
                    print(f"GPU {device} cached: {torch.cuda.memory_reserved(device)}")
                    print()

                elif self.verbose.lower() == "detail":
                    print(torch.cuda.memory_summary())
                    print()

            self.inference(self.model)
            self.destroy()

        except BaseException:
            traceback.print_exc()
            self.destroy()
