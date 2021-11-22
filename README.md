<p align="center">
  <img src="https://user-images.githubusercontent.com/38183241/125905410-1ee984a3-c5a9-4d8c-ba40-46fca740f514.png" width=380>
</p>
<p align="center">
<a href="https://github.com/tunib-ai/parallelformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/tunib-ai/parallelformers.svg" /></a> <a href="https://github.com/tunib-ai/parallelformers/blob/master/LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/></a> <a href="https://tunib-ai.github.io/parallelformers"><img alt="Docs" src="https://img.shields.io/badge/docs-passing-success.svg"/></a> <a href="https://github.com/tunib-ai/parallelformers/issues"><img alt="Issues" src="https://img.shields.io/github/issues/tunib-ai/parallelformers"/></a>

</p>
<br>


- Parallelformers, which is based on [Megatron LM](https://github.com/NVIDIA/Megatron-LM), is designed to make model parallelization easier.
- You can parallelize various models in [HuggingFace Transformers](https://github.com/huggingface/transformers) on multiple GPUs with **a single line of code.**
- Currently, Parallelformers **only supports inference**. Training features are NOT included.

<br>


### What's New:
* October 24, 2021 [Docker support](https://github.com/tunib-ai/parallelformers#are-you-getting-some-errors-in-docker-container).
* July 28, 2021 [Released a tech blog](https://tunib.tistory.com/entry/Parallelformers-Journey-to-deploying-big-modelsTUNiB?category=899987).
* July 18, 2021 [Released Parallelformers 1.0](https://github.com/tunib-ai/parallelformers/releases/tag/1.0).

<br>

## Why Parallelformers?
You can load a model that is too large for a single GPU. For example, using Parallelformers, you can load a model of 12GB on two 8 GB GPUs. In addition, you can save your precious money because usually multiple smaller size GPUs are less costly than a single larger size GPU.

## Installation
Parallelformers can be easily installed using the `pip` package manager. All the dependencies such as [torch](https://pypi.org/project/torch/), [transformers](https://pypi.org/project/transformers/), and [dacite](https://pypi.org/project/dacite/) should be installed automatically with the following command. Be careful that the name is plural.
```console
pip install parallelformers
```

## Getting Started
#### 1. Create a HuggingFace transformers model. 
You don't need to call `.half()` or `.cuda()` as those functions will be invoked automatically. It is more memory efficient to start parallelization on the CPU.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
```

#### 2. Put the `model` in the `parallelize()` function.
```python
from parallelformers import parallelize

parallelize(model, num_gpus=2, fp16=True, verbose='detail')
```

Since `nvidia-smi` shows the reserved cache area, it is difficult to check the exact allocated memory. To check the allocated memory state well, **you can set the verbose option as `'detail'` or `'simple'`.** (default is `None`)

```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |    2721 MB |    2967 MB |    2967 MB |  251905 KB |
|       from large pool |    2720 MB |    2966 MB |    2966 MB |  251904 KB |
|       from small pool |       1 MB |       1 MB |       1 MB |       1 KB |
|---------------------------------------------------------------------------|

GPU:0 => 2.72GB
```
```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 1                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |    2721 MB |    2967 MB |    2967 MB |  251905 KB |
|       from large pool |    2720 MB |    2966 MB |    2966 MB |  251904 KB |
|       from small pool |       1 MB |       1 MB |       1 MB |       1 KB |
|---------------------------------------------------------------------------|

GPU:1 => 2.72GB
```

#### 3. Do Inference as usual. 
You don't have to call `.cuda()` when creating input tokens. **Note that you should input both input tokens and attention masks to the model.** (`**inputs` is the recommended way for this)
```python
inputs = tokenizer("Parallelformers is", return_tensors="pt")

outputs = model.generate(
    **inputs,
    num_beams=5,
    no_repeat_ngram_size=4,
    max_length=15,
)

print(f"Output: {tokenizer.batch_decode(outputs)[0]}")
``` 
```
Output: Parallelformers is an open-source library for parallel programming ...
```

#### 4. Deploy the model to the server as usual. 
The parallelization process does not affect the web server because they are automatically synchronized.
```python
from flask import Flask

app = Flask(__name__)


@app.route("/generate_text/<text>")
def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        num_beams=5,
        no_repeat_ngram_size=4,
        max_length=15,
    )
    
    outputs = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )
    
    return {
        "inputs": text,
        "outputs": outputs[0],
    }


app.run(host="0.0.0.0", port=5000)
```

You can send a request to the web server as follows:
```
$ curl -X get "YOUR_IP:5000/generate_text/Messi"
```
And the following result should be returned.
```
{"inputs": "Messi", "outputs": "Messi is the best player in the world right now. He is the"}
```

#### 5. Check the current GPU states.
You can check GPU states using `.memory_allocated()`, `.memory_reserved()` and `.memory_chached()` to make sure the parallelization is successful.
```python
model.memory_allocated()
model.memory_reserved()
model.memory_chached()
```
```
{'cuda:0':XXXXXX, 'cuda:1':XXXXXX}
```

#### 6. Manage the model parallelization states.
You can manage model parallelization states using `.cuda()`, `.cpu()` and `.to()`. **The model parallelization process ends if you call those functions.**
```python
model.cuda()

print(torch.cuda.memory_summary(0))
print(torch.cuda.memory_summary(1))
```
Check the allocated memory status using `torch.cuda.memory_summary()`.
```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |    5121 MB |    5121 MB |    5121 MB |    1024 B  |
|       from large pool |    5120 MB |    5120 MB |    5120 MB |       0 B  |
|       from small pool |       1 MB |       1 MB |       1 MB |    1024 B  |
|---------------------------------------------------------------------------|

GPU0 => 5.12GB
```
```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 1                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |       0 B  |    1024 B  |    1024 B  |    1024 B  |
|       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |
|       from small pool |       0 B  |    1024 B  |    1024 B  |    1024 B  |
|---------------------------------------------------------------------------|

GPU1 => 0.00GB
```
If you switch to the CPU mode, it works like this.
```python
model.cpu()

print(torch.cuda.memory_summary(0))
print(torch.cuda.memory_summary(1))
```
```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |       0 B  |    5121 MB |    5121 MB |    5121 MB |
|       from large pool |       0 B  |    5120 MB |    5120 MB |    5120 MB |
|       from small pool |       0 B  |       1 MB |       1 MB |       1 MB |
|---------------------------------------------------------------------------|

GPU0 => 0.00GB
```
```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 1                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |       0 B  |    1024 B  |    1024 B  |    1024 B  |
|       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |
|       from small pool |       0 B  |    1024 B  |    1024 B  |    1024 B  |
|---------------------------------------------------------------------------|

GPU1 => 0.00GB
```

## Are you getting some errors in docker container?
I recently found out that ALL errors that occur in environments with limited resources such as docker containers are due to **shared memory size**. So, if you want to use larger models with parallelformers in docker containers, **INCREASE the size of shared memory by `--shm-size=?gb` or REMOVE the limitation of shared memory size by `--ipc=host`**. the larger shared memory size is required if you want to use larger model.

## Supported Models
Currently, most models in Huggingface transformers are supported. All layers in the models listed below can be parallelized.
They include vision models like `ViT`, `CLIP` and speech models like `Wav2Vec2` as well as language models.

<details>
  <summary>Fully Supported Models</summary>

* ALBERT
* BART
* BARThez (=BERT)
* BERT
* BERTweet (=BERT)
* BertJapanese (=BERT)
* BertGeneration
* Blenderbot
* Blenderbot Samll
* BORT (=BERT)
* CamemBERT (=RoBERTa)
* CLIP
* CPM
* CTRL
* DeBERTa
* DeBERTa-v2
* DeiT
* DETR
* DialoGPT (=GPT2)
* DistilBERT
* DPR (=BERT)
* ELECTRA
* FlauBERT (=XLM)
* FSMT
* Funnel Transformer
* herBERT (=RoBERTa)
* I-BERT
* LayoutLM
* LED
* Longformer
* LUKE
* LXMERT
* MarianMT
* M2M100
* MBart
* Mobile BERT
* MPNet
* MT5 (=T5)
* Megatron BERT (=BERT)
* Megatron GPT2 (=GPT2)
* OpenAI GPT
* OpenAI GPT2
* GPTNeo
* Hubert
* Pegasus
* PhoBERT (=RoBERTa)
* Reformer
* RetriBERT
* RoBERTa
* RoFormer
* Speech2Text
* T5
* ByT5 (=T5)
* TAPAS
* TransformerXL
* ViT
* VisualBERT
* Wav2Vec2
* XLM
* XLM-RoBERTa (=RoBERTa)
* XLNet
* XLSR-Wave2Vec2
  
</details>


At present the following models are [partly supported or not supported](FAQ.md#q-why-are-some-models-not-supported). 

<details> 
  <summary>Partly Supported Models</summary>

* BigBird 
* BigBirdPegasus
* ConvBERT
* ProphetNet 
* XLM-ProphetNet

</details>

<details> 
  <summary>Unsupported Models</summary>

* SqueezeBERT
* RAG
  
</details>

## Advanced Usage
Refer to [POLICY.md](POLICY.md)

## FAQ
Refer to [FAQ.md](FAQ.md).

## Contributing
Refer to [CONTRIBUTING.md](CONTRIBUTING.md)

## Documentation
For more detailed information, see [full documentation](https://tunib-ai.github.io/parallelformers/)

## Citation
If you find this library useful, please consider citing:

```
@misc{parallelformers,
  author       = {Ko, Hyunwoong},
  title        = {Parallelformers: An Efficient Model Parallelization Toolkit for Deployment},
  howpublished = {\url{https://github.com/tunib-ai/parallelformers}},
  year         = {2021},
}
```

## LICENSE
`Parallelformers` is licensed under the terms of the Apache License 2.0.

Copyright 2021 TUNiB inc. http://www.tunib.ai. All Rights Reserved.
