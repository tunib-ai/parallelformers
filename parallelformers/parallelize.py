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
import traceback
from contextlib import suppress
from dataclasses import _is_dataclass_instance
from typing import Any, Dict

import torch
import torch.multiprocessing as mp
from dacite import Config, from_dict
from torch import nn
from transformers.file_utils import ModelOutput

from parallelformers.parallel.process import ParallelProcess
from parallelformers.utils import rgetattr, rsetattr


class parallelize(object):
    """
    Parallelformers end-point function

    Args:
        model (nn.Module): Huggingface pre-trained transformer model.
        fp16: (bool): whether use FP16 or not.
        num_gpus (int): number of GPU for parallelization.
        master_addr (str): master process address for process communication (default='127.0.0.1')
        master_port (int): master process port for process communication (default=29500)
        backend (str): distributed backend (default='nccl')
        verbose (str): logging current gpu states (one of ['detail', 'simple', None]
        init_method (str): multiprocess initialization method. (It is safe to set `init_method` to `spawn`.)
        daemon (bool): whether make process daemon or not (default=True)

    Notes:
        We want to use this object as a simple function rather than a class.
        So we broke the PEP8 rules and set class names that start with a lowercase letter.

    Examples:
        >>> # 1. Import Huggingface and Parallelformers modules
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from parallelformers import parallelize

        >>> # 2. Create Huggingface model and tokenizer.
        >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

        >>> # 3. Parallelize using Parallelformers.
        >>> parallelize(model, num_gpus=4, fp16=True)

        >>> # 4. Do inference as usual.
        >>> inputs = tokenizer("Parallelformers is", return_tensors="pt")
        >>> outputs = model.generate(**inputs, num_beams=5, no_repeat_ngram_size=4)
        >>> print(f"Output: {tokenizer.batch_decode(outputs)[0]}")
        'Output: Parallelformers is an open-source library for parallel programming ...'

    """

    def __init__(
        self,
        model: nn.Module,
        fp16: bool,
        num_gpus: int,
        custom_policies=None,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        backend="nccl",
        verbose: str = None,
        init_method: str = "spawn",
        daemon: bool = True,
        seed: int = None,
    ):
        self.init_environments(
            num_gpus,
            master_addr,
            master_port,
        )

        # Using fork often leads to deadlock.
        if mp.get_start_method() != init_method:
            with suppress(Exception):
                mp.set_start_method(init_method, force=True)

        for param in model.parameters():
            param.requires_grad = False
            param.detach()

        self.preprocess_for_wav2vec(model)
        self.model = model.half() if fp16 else model
        self.model.eval()
        self.fp16 = fp16
        self.num_gpus = num_gpus
        self.backend = backend
        self.daemon = daemon
        self.verbose = verbose
        self.seed = seed
        self.custom_policies = custom_policies

        self.processes = []
        self.parallel_mutexes = []
        self.inference_mutexes = []
        self.inputs_queues = []
        self.outputs_queues = []
        self.orig_methods = {}

        hijack_methods = ["generate", "forward", "to", "cpu", "cuda"]

        for attr in hijack_methods:
            if hasattr(self.model, attr):
                fn = getattr(self.model, attr)
                self.orig_methods[attr] = fn

        self.parallelize()
        for attr in hijack_methods:
            if hasattr(self.model, attr):
                self.register_hijack_methods(attr)

        self._memory_logger = ["memory_reserved", "memory_allocated", "memory_cached"]

        for attr in self._memory_logger:
            self.register_memory_methods(attr)

    def preprocess_for_wav2vec(self, model: nn.Module) -> None:
        """
        There is one missing parameter in the Huggingface Wav2Vec model,
        and the user loses control over this parameter, making parallelization impossible.

        To use multiprocessing, if `requries_grad` is `True`, it must be a leaf tensor,
        but this tensor (`conv.weight`) does not follow this rule, so multiprocessing becomes impossible.
        Therefore, we detach this parameter to enable multiprocessing.

        Args:
            model (nn.Module): model weight
        """

        with suppress(Exception):
            from transformers.models.hubert.modeling_hubert import (
                HubertPositionalConvEmbedding,
            )
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2PositionalConvEmbedding,
            )

            for child in model.named_children():
                layer_object = child[1]

                if layer_object.__class__ in [
                    Wav2Vec2PositionalConvEmbedding,
                    HubertPositionalConvEmbedding,
                ]:
                    detached_parameter = rgetattr(layer_object, "conv.weight").detach()
                    rsetattr(layer_object, "conv.weight", detached_parameter)

                else:
                    self.preprocess_for_wav2vec(layer_object)

    def init_environments(
        self,
        num_gpus: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        """
        Initialize environment variables

        Args:
            num_gpus (int): number of GPU for parallelization.
            master_addr (str): master process address for process communication
            master_port (int): master process port for process communication
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(num_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(
            [str(i) for i in range(num_gpus)]
        )

    def register_hijack_methods(self, method: str) -> None:
        """
        Intercept the flow by changing some methods (e.g. forward, generate, ...)
        in the model to `self.hijack` methods.

        Args:
            method (str): name of method
        """

        setattr(
            self.model,
            method,
            lambda *inputs, **kwargs: self.hijack(
                inputs=inputs,
                kwargs=kwargs,
                func=method,
            ),
        )

    def register_memory_methods(self, method: str) -> None:
        """
        Add several methods to check GPU occupancy status of models located in other process.

        Args:
            method (str): name of method
        """

        setattr(
            self.model,
            method,
            lambda: self.hijack(
                inputs="dummy",
                kwargs={"dummy": "dummy"},
                func=method,
            ),
        )

    def deparallelize(self) -> None:
        """
        Remove all methods registered in the model
        and join all GPU processes to main process.
        """

        if hasattr(self, "orig_methods"):
            for k, v in self.orig_methods.items():
                setattr(self.model, k, v)

        if hasattr(self, "_memory_logger"):
            for method in self._memory_logger:
                setattr(self.model, method, None)

        if hasattr(self, "processes"):
            for process in self.processes:
                process.join()

        torch.cuda.empty_cache()

    @torch.no_grad()
    def parallelize(self) -> None:
        """Create processes for model parallelization and parallel inference"""

        try:
            for rank in range(self.num_gpus):
                parallel_mutex = mp.Event()
                inference_mutex = mp.Event()
                self.parallel_mutexes.append(parallel_mutex)
                self.inference_mutexes.append(inference_mutex)

                inputs_queue = mp.Queue()
                outputs_queue = mp.Queue()
                self.inputs_queues.append(inputs_queue)
                self.outputs_queues.append(outputs_queue)

                process = ParallelProcess(
                    rank=rank,
                    model=self.model,
                    fp16=self.fp16,
                    num_gpus=self.num_gpus,
                    inputs_queue=inputs_queue,
                    outputs_queue=outputs_queue,
                    parallel_mutex=parallel_mutex,
                    inference_mutex=inference_mutex,
                    backend=self.backend,
                    verbose=self.verbose,
                    custom_policies=self.custom_policies,
                    seed=self.seed,
                )

                process.daemon = self.daemon
                # When the main process done, all processes should frees resources.
                # So default value is True, but change it according to your needs.

                process.start()
                self.processes.append(process)

            for p_mutex in self.parallel_mutexes:
                p_mutex.wait()

        except BaseException:
            traceback.print_exc()
            self.deparallelize()

    @staticmethod
    def _deallocate(item):
        if torch.is_tensor(item) and item.is_cuda:
            item.cpu()

        elif isinstance(item, list) or isinstance(item, tuple):
            for i in item:
                if torch.is_tensor(i) and i.is_cuda:
                    i.cpu()

        elif isinstance(item, dict):
            for i in item:
                if torch.is_tensor(item[i]) and item[i].is_cuda:
                    item[i].cpu()

        return item

    @torch.no_grad()
    def hijack(
        self,
        inputs: Any,
        kwargs: Dict,
        func: str,
    ) -> Any:
        """
        Transfers the input passed to the main process to another process,
        and transfers the output to the main process and outputs it to the user.

        Args:
            inputs (Any): inputs of model
            kwargs (Dict): arguments of model
            func (str): name of method that hijacked

        Returns:
            Any: outputs of model
        """
        try:
            for i_mutex, i_queue in zip(
                self.inference_mutexes,
                self.inputs_queues,
            ):
                inputs = self._deallocate(inputs)

                for k in kwargs:
                    kwargs[k] = self._deallocate(kwargs[k])

                i_queue.put((inputs, kwargs, func))
                i_mutex.set()
                # producer part

            if func in ["to", "cpu", "cuda"]:
                self.deparallelize()

                if func == "cpu":
                    self.model = self.model.cpu(*inputs, **kwargs)
                elif func == "cuda":
                    self.model = self.model.cuda(*inputs, **kwargs)
                else:
                    self.model = self.model.to(*inputs, **kwargs)

                return self.model
            else:
                outputs = []
                for o_queue in self.outputs_queues:
                    output = o_queue.get()
                    outputs.append(output)
                    # consumer part

                if func in self._memory_logger:
                    final_output = dict(outputs)
                else:
                    final_output = outputs[0]

                # non-picklable object to original dataclass
                if (
                    isinstance(final_output, dict)
                    and "orig_dataclass_type" in final_output
                ):
                    orig_dataclass_type = final_output["orig_dataclass_type"]
                    del final_output["orig_dataclass_type"]

                    final_output = from_dict(
                        orig_dataclass_type,
                        final_output,
                        config=Config(check_types=False),
                    )

                return final_output

        except BaseException:
            traceback.print_exc()
            self.deparallelize()
