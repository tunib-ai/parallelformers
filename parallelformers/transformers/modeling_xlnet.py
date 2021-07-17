# Copyright 2021 Huggingface & TUNiB inc.
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
from torch import distributed as dist
from transformers.models.xlnet.modeling_xlnet import XLNetRelativeAttention


class XLNetRelativeAttention_(XLNetRelativeAttention):
    """
    Fixed (5 lines):
        Add Allreduce operation
    """

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:

            dist.all_reduce(
                attn_out,
                group=self.mp_group,
            )

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output
