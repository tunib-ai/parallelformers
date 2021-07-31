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
"""
After test_XXX_model.sh, we can check similarity of layer-wise output hidden states.
Parallelformers guarantees that the results before and after parallelization are almost 100% identical.

examples:
    $ test_bert.model.sh
    $ python compute_similarity.py --path="bert"

layer: 0
cosine similarity (per token):  tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.]], device='cuda:0')
cosine similarity (mean):  tensor(1., device='cuda:0')

layer: 1
cosine similarity (per token):  tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]],
       device='cuda:0')
cosine similarity (mean):  tensor(1., device='cuda:0')

...

layer: 12
cosine similarity (per token):  tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]],
       device='cuda:0')
cosine similarity (mean):  tensor(1., device='cuda:0')
"""

import os
from argparse import ArgumentParser

import torch

parser = ArgumentParser()
parser.add_argument("--path", required=True)
args = parser.parse_args()

pf = torch.load(os.path.join(args.path, "output-pf-fp32.pt"))
non_pf = torch.load(os.path.join(args.path, "output-non-pf-fp32.pt"))
# pf means 'parallelformers'

for i, (p, n) in enumerate(zip(pf, non_pf)):
    cos = torch.cosine_similarity(p.cuda(), n.cuda(), dim=-1, eps=1e-12)
    print(f"layer: {i}")
    print("cosine similarity (per token): ", cos)
    print("cosine similarity (mean): ", cos.mean())
    print()
