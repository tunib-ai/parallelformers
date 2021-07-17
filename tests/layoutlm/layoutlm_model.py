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
from transformers import AutoModel, AutoTokenizer

from parallelformers import parallelize


class TestModel(unittest.TestCase):
    @torch.no_grad()
    def test_forward(self, model, use_pf, fp16):
        words = ["Hello", "world"]
        normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        token_boxes = []
        for word, box in zip(words, normalized_word_boxes):
            word_tokens = tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = tokenizer(" ".join(words), return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        bbox = torch.tensor([token_boxes])

        if not use_pf:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            bbox = bbox.cuda()
            token_type_ids = token_type_ids.cuda()

        output = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        print("forward:", output.hidden_states[0][0])
        torch.save(
            output.hidden_states,
            f"output-{'pf' if use_pf else 'non-pf'}-{'fp16' if fp16 else 'fp32'}.pt",
        )
        assert isinstance(output.hidden_states[0][0], torch.Tensor)


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

    model = AutoModel.from_pretrained(args.name).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.name)
    print(f"Test Name: [{model.__class__.__name__}]-[{args.test_name}]\n")

    gpus = [
        _
        for _ in range(
            args.gpu_from,
            args.gpu_to + 1,
        )
    ]

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

        for i in gpus:
            print(f"GPU {i} alloc: {torch.cuda.memory_allocated(i)}")
            print(f"GPU {i} cached: { torch.cuda.memory_reserved(i)}")
            print()

    test = TestModel()
    test.test_forward(model, args.use_pf, args.fp16)
    print("=========================================================")
