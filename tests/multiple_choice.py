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
import unittest
from argparse import ArgumentParser

import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from parallelformers import parallelize


class TestForMultipleChoice(unittest.TestCase):
    @torch.no_grad()
    def test_forward(self, model, tokens, labels):
        output = model(
            **{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels
        ).logits  # batch size is 1
        print("forward:", output)
        print()
        assert isinstance(output, torch.Tensor)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = ArgumentParser()
    parser.add_argument("--test-name", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu-from", required=True, type=int)
    parser.add_argument("--gpu-to", required=True, type=int)
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--use-pf", default=False, action="store_true")
    args = parser.parse_args()

    model = AutoModelForMultipleChoice.from_pretrained(args.name).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.name)
    print(f"Test Name: [{model.__class__.__name__}]-[{args.test_name}]\n")

    gpus = [
        _
        for _ in range(
            args.gpu_from,
            args.gpu_to + 1,
        )
    ]

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    labels = torch.tensor(0).unsqueeze(
        0
    )  # choice0 is correct (according to Wikipedia ;)), batch size 1

    encoding = tokenizer(
        [[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True
    )

    if args.use_pf:
        parallelize(
            model,
            num_gpus=args.gpu_to + 1,
            fp16=args.fp16,
            verbose="simple",
        )
    else:
        if args.fp16:
            model = model.half()

        model = model.cuda()
        # import pdb;pdb.set_trace()
        for e in encoding:
            if torch.is_tensor(encoding[e]):
                encoding[e] = encoding[e].cuda()
        labels = labels.cuda()
        print("labels : ", labels)
        for i in gpus:
            print(f"GPU {i} alloc: {torch.cuda.memory_allocated(i)}")
            print(f"GPU {i} cached: { torch.cuda.memory_reserved(i)}")
            print()

    test = TestForMultipleChoice()
    test.test_forward(model, encoding, labels)
    print("=========================================================")
