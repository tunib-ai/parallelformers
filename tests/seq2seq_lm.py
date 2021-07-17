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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from parallelformers import parallelize


class TestForCausalLM(unittest.TestCase):
    @torch.no_grad()
    def test_generation(self, model, tokens, tokenizer):
        output = model.generate(
            **tokens,
            num_beams=6,
            max_length=40,
            no_repeat_ngram_size=4,
        )

        gen = tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
        )[0]
        print("generate:", gen.replace("\n", "\\n"))
        print()
        assert isinstance(gen, str)

    @torch.no_grad()
    def test_forward(self, model, tokens, use_pf, fp16):
        output = model(
            **tokens,
            decoder_input_ids=tokens["input_ids"],
            output_hidden_states=True,
        )
        torch.save(
            output.encoder_hidden_states,
            f"output-encoder-{'pf' if use_pf else 'non-pf'}-{'fp16' if fp16 else 'fp32'}.pt",
        )
        torch.save(
            output.decoder_hidden_states,
            f"output-decoder-{'pf' if use_pf else 'non-pf'}-{'fp16' if fp16 else 'fp32'}.pt",
        )
        print()


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

    model = AutoModelForSeq2SeqLM.from_pretrained(args.name).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.name)
    print(f"Test Name: [{model.__class__.__name__}]-[{args.test_name}]\n")

    gpus = [
        _
        for _ in range(
            args.gpu_from,
            args.gpu_to + 1,
        )
    ]

    tokens = tokenizer(
        "Kevin is <mask> man in the <mask>. Today He is working with his friend.",
        return_tensors="pt",
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
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].cuda()

        for i in gpus:
            print(f"GPU {i} alloc: {torch.cuda.memory_allocated(i)}")
            print(f"GPU {i} cached: { torch.cuda.memory_reserved(i)}")
            print()

    test = TestForCausalLM()
    test.test_generation(model, tokens, tokenizer)
    test.test_forward(model, tokens, use_pf=args.use_pf, fp16=args.fp16)
    print("=========================================================")
