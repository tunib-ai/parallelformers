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
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch.nn as nn
from torch import Size, Tensor

from parallelformers.parallel.slicing import TensorSlicer
from parallelformers.policies.base import Layer, Policy
from parallelformers.policies.base.auto import AutoPolicy
from parallelformers.utils import rgetattr, rhasattr, rsetattr


class TensorReplacer(object):
    r"""
    Replace original Huggingface's layer into Megatron tensor sliced layer.

    Args:
        model (nn.Module): Huggingface pre-trained transformer model
        mp_group (Any): process group for model parallelism
        fp16: (bool): Whether use FP16 or not.
        num_gpus (int): number of GPUs
        custom_policies (Union[Policy, List[Policy]]): custom policy object (default=None)
    """

    def __init__(
        self,
        model: nn.Module,
        mp_group: Any,
        fp16: bool,
        num_gpus: int,
        custom_policies: Union[Policy, List[Policy]],
    ) -> None:
        self.model = model
        self.config = model.config
        self.fp16 = fp16
        self.num_gpus = num_gpus
        self.mp_group = mp_group
        self.varname = "layer"
        self.slicer = TensorSlicer(self.mp_group)

        if isinstance(custom_policies, Iterable):
            self.policies = custom_policies
        elif isinstance(custom_policies, Policy):
            self.policies = [custom_policies]
        else:
            self.policies = None

    def auto_policy(self) -> Optional[List[Policy]]:
        """Find the proper policy for current model using AutoPolicy"""

        auto = AutoPolicy()
        policy_cls = auto.get_policy(self.model)

        assert policy_cls is not None, (
            f"{self.model.__class__.__qualname__} is not supported yet.\n"
            f"Currently we support {[i.__qualname__ for i in auto.available().keys()]}.\n"
            f"To apply to unsupported models, you need to create a custom Policy object."
        )

        return policy_cls

    def replace_modules(self) -> None:
        """Replace original huggingface layers to Megtraon tensor sliced layers"""
        if self.policies is None:
            self.policies = self.auto_policy()

        for policy in self.policies:
            self.replace_user_define_modules(self.model, policy)
            self.replace_orig_to_megatron_modules(self.model, policy)

    def replace_user_define_modules(
        self,
        model: nn.Module,
        policy_cls: Type[Policy],
    ) -> None:
        """
        Replace modules in the model by user defined policy

        Args:
            model (nn.Module): model weight
            policy_cls (Type[Policy]): class of policy
        """
        for _, child in model.named_children():
            if child.__class__ == nn.ModuleList:
                child = child[0]

            replace_modules = policy_cls.replace_modules()

            if child.__class__.__qualname__ in replace_modules.keys():
                for cls_name, cls in replace_modules.items():
                    if child.__class__.__qualname__ == cls_name:
                        for key in cls.__dict__.keys():
                            rsetattr(
                                child.__class__,
                                "mp_group",
                                self.mp_group,
                            )

                            if rhasattr(child.__class__, key):
                                rsetattr(
                                    child.__class__,
                                    key,
                                    rgetattr(cls, key),
                                )

            self.replace_user_define_modules(child, policy_cls)

    def replace_orig_to_megatron_modules(
        self,
        model: nn.Module,
        policy_cls: Type[Policy],
    ) -> nn.Module:
        """
        Replace original Huggingface layers to Megatron tensor sliced layers

        Args:
            model (nn.Module): model weight
            policy_cls (Type[Policy]): class of policy

        Returns:
            nn.Module: parallelized paramerters
        """
        for name, child in model.named_children():
            if child.__class__ == policy_cls.original_layer_class():
                policy = policy_cls(layer=child)
                arguments = policy.replace_arguments(self.config, self.num_gpus)

                for k, v in arguments.items():
                    with suppress(Exception):
                        rsetattr(policy, f"{self.varname}.{k}", v)

                rsetattr(model, name, self.make_megatron_layer(policy))

            self.replace_orig_to_megatron_modules(child, policy_cls)

        return model

    def preprocess(
        self,
        function_output: List[Layer],
        policy: Policy,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Preprocess user's policy object to replace tensors

        Args:
            function_output (List[Layer]): list of layers in the policy object
            policy (Policy): policy object

        Returns:
            Tuple[Dict, Dict, Dict, Dict]:
                Tuple of dictionaries of parameters and attributes required for tensor slicing
        """
        weight_dict, bias_dict, weight_attr_dict, bias_attr_dict = {}, {}, {}, {}

        for layer_params in function_output:
            w = layer_params.weight
            b = layer_params.bias
            replace = layer_params.replace
            n_fused = layer_params.n_fused
            reversed = layer_params.reversed
            ignore = layer_params.ignore_checker

            if w is not None:
                if rhasattr(policy, f"{self.varname}.{w}"):
                    w_layer = rgetattr(policy, f"{self.varname}.{w}")
                    weight_dict[f"{self.varname}.{w}"] = w_layer
                    weight_attr_dict[f"{self.varname}.{w}"] = (
                        n_fused,
                        reversed,
                    )

                    orig_layer_name = ".".join(w.split(".")[:-1])
                    orig_layer = rgetattr(
                        policy,
                        f"{self.varname}.{orig_layer_name}",
                    )
                elif not ignore:
                    raise Exception(
                        f"'{policy.original_layer_class().__qualname__}' object has no attribute '{w}'"
                    )

            if b is not None:
                if rhasattr(policy, f"{self.varname}.{b}"):
                    b_layer = rgetattr(policy, f"{self.varname}.{b}")
                    bias_dict[f"{self.varname}.{b}"] = b_layer
                    bias_attr_dict[f"{self.varname}.{b}"] = (
                        n_fused,
                        reversed,
                    )

                    orig_layer_name = ".".join(b.split(".")[:-1])
                    orig_layer = rgetattr(
                        policy,
                        f"{self.varname}.{orig_layer_name}",
                    )
                elif not ignore:
                    raise Exception(
                        f"'{policy.original_layer_class().__qualname__}' object has no attribute '{b}'"
                    )

            if not w and not b:
                raise Exception("both weight and bias are empty !")

            if replace is not None:
                orig_layer.__class__ = replace
                orig_layer.mp_group = self.mp_group

        return weight_dict, bias_dict, weight_attr_dict, bias_attr_dict

    def set_parameters(
        self,
        policy: Policy,
        weight_name: Dict[str, Tensor],
        bias_name: Dict[str, Tensor],
        weight_param: Dict[str, Tensor],
        bias_param: Dict[str, Tensor],
        suffix: str = "data",
    ) -> Policy:
        """
        Set sliced parameters into original model

        Args:
            policy (Policy): policy object
            weight_name (Tuple[str]): names of layer's weight
            bias_name (Tuple[str]): names of layer's bias
            weight_param (Tuple[Tensor]): parameters of sliced weight
            bias_param (Tuple[Tensor]): parameters of sliced bias
            suffix (str): name of suffix in the parameters

        Returns:
            Policy: policy object
        """
        for name, param in zip(weight_name, weight_param):
            rsetattr(policy, f"{name}.{suffix}", param)
            self.set_layer_size(policy, name, param.size())

        for name, param in zip(bias_name, bias_param):
            rsetattr(policy, f"{name}.{suffix}", param)

        return policy

    @staticmethod
    def set_layer_size(
        policy: Policy,
        name: str,
        size: Size,
    ) -> None:
        """
        Apply resize layer size to original layer object

        Args:
            policy (Policy): policy object
            name (str): name of parameters
            size (Size): size of resized parameters
        """
        layer_name = ".".join(f"{name}".split(".")[:-1])
        if rhasattr(policy, f"{layer_name}.nf"):
            rsetattr(
                policy,
                f"{layer_name}.nf",
                size[1],
            )
        else:
            for name in ["channels", "features"]:
                if name == "channels":
                    direction = ["in", "out"]
                else:
                    direction = ["out", "in"]
                for i, direction in enumerate(direction):
                    if rhasattr(policy, f"{layer_name}.{direction}_{name}"):
                        rsetattr(
                            policy,
                            f"{layer_name}.{direction}_{name}",
                            size[i],
                        )

    def make_megatron_layer(self, policy: Policy) -> nn.Module:
        """
        Make Megatron tensor sliced layers from original Huggingface layers by tensor slicing.

        Args:
            policy (Policy): policy object

        Returns:
            nn.Module: sliced model layer
        """
        attn_qkvw, attn_qkvb, attn_qkvw_attr, attn_qkvb_attr = self.preprocess(
            policy.attn_qkv(),
            policy,
        )
        attn_outw, attn_outb, attn_outw_attr, attn_outb_attr = self.preprocess(
            policy.attn_out(),
            policy,
        )
        mlp_inw, mlp_inb, mlp_inw_attr, mlp_inb_attr = self.preprocess(
            policy.mlp_in(),
            policy,
        )
        mlp_outw, mlp_outb, mlp_outw_attr, mlp_outb_attr = self.preprocess(
            policy.mlp_out(),
            policy,
        )

        policy = self.set_parameters(
            policy,
            attn_qkvw,
            attn_qkvb,
            *self.slicer.column_slice(
                (attn_qkvw, attn_qkvb),
                (attn_qkvw_attr, attn_qkvb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            attn_outw,
            attn_outb,
            *self.slicer.row_slice(
                (attn_outw, attn_outb),
                (attn_outw_attr, attn_outb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            mlp_inw,
            mlp_inb,
            *self.slicer.column_slice(
                (mlp_inw, mlp_inb),
                (mlp_inw_attr, mlp_inb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            mlp_outw,
            mlp_outb,
            *self.slicer.row_slice(
                (mlp_outw, mlp_outb),
                (mlp_outw_attr, mlp_outb_attr),
            ),
        )

        return policy.layer
