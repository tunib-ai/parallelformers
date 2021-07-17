<p align="center">
  <img src="https://user-images.githubusercontent.com/38183241/125905410-1ee984a3-c5a9-4d8c-ba40-46fca740f514.png" width=380>
</p>
<p align="center">
<a href="https://github.com/tunib-ai/parallelformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/tunib-ai/parallelformers.svg" /></a> <a href="https://github.com/tunib-ai/parallelformers/blob/master/LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" /></a> <a href="https://github.com/tunib-ai/parallelformers/issues"><img alt="Issues" src="https://img.shields.io/github/issues/tunib-ai/parallelformers" /></a>

</p>
<br>

- Parallelformers, which is based on [Megatron LM](https://github.com/NVIDIA/Megatron-LM), is designed to make model parallelization easier.
- You can parallelize various models in [HuggingFace Transformers](https://github.com/huggingface/transformers) on multiple GPUs with **a single line of code.**
- Currently, Parallelformers **only supports inference**. Training features are NOT included.

<br>

### Disclaimer
The logo of Parallelformers is adapted from the [Hugging Face emoji](https://emojipedia.org/hugging-face/). It has nothing to do with [Hugging Face](https://huggingface.co/), the famous NLP start-up.


## Why Parallelformers?
You can load a model that is too large for a single GPU. For example, using Parallelformers, you can load a model of 12GB on two 8 GB GPUs. In addition, you can save your precious money because usually multiple smaller size GPUs are less costly than a single larger size GPU.

## Installation
Parallelformers can be easily installed using the `pip` package manager. All the dependencies such as [torch](https://pypi.org/project/torch/), [transformers](https://pypi.org/project/transformers/), and [dacite](https://pypi.org/project/dacite/) should be installed automatically with the following command. Be careful that the name is plural.
```console
pip install parallelformers
```


## Getting Started
1. Create a HuggingFace transformers model. You don't need to call `.half()` or `.cuda()` as those functions will be invoked automatically. It is more memory efficient to start parallelization on the CPU.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
```


2. Put the `model` in the `parallelize()` function.
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

3. Do Inference as usual. You don't have to call `.cuda()` when creating input tokens. **Note that you should input both input tokens and attention masks to the model.** (`**inputs` is the recommended way for this)
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

Why Haskell??? It's written in Python... Ì†æÌ¥£
```
Output: Parallelformers is an open-docs library for parallel programming in Haskell
```

4. Deploy the model to the server as usual. The parallelization process does not affect the web server because they are automatically synchronized.
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

5. Check the current GPU states using `.memory_allocated()`, `.memory_reserved()` and `.memory_chached()` to make sure the parallelization is successful.
```python
model.memory_allocated()
model.memory_reserved()
model.memory_chached()
```
```
{'cuda0':XXXXXX, 'cuda1':XXXXXX}
```

6. Manage the model parallelization states using `.cuda()`, `.cpu()` and `.to()`. **The model parallelization process ends if you call those functions.**
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


## Supported Models
Currently, most models in Huggingface transformers are supported. All layers in the models listed below can be parallelized.
They include vision models like `ViT`, `CLIP` and speech models like `Wav2Vec2` as well as language models. Ì†ºÌæâÌ†ºÌæâ

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

At present the following models are not supported or partly supported.
* BigBird
* BigBirdPegasus
* ConvBERT
* SqueezeBERT
* ProphetNet
* XLM-ProphetNet
* RAG



## `Policy` Class

In Parallelformers, every model has its own `Policy` class that manages the overall parallelization configurations. (Check [this](https://github.com/tunib-ai/parallelformers-dev/tree/main/parallelformers/policies).) In most cases, you don't have to care about them because the policies of most Hugging Face models are pre-defined in the [`AutoPolicy`](https://github.com/tunib-ai/parallelformers-dev/blob/main/parallelformers/policies/base/auto.py#L180) class. If you want to use a new model that is not in the `AutoPolicy`, you need to add a `Policy` class for yourself. Below are the basic syntax related to the `Policy` class.


#### 5.1.1. `Layer` Class
Most methods in the `Policy` class return a list of `Layer` classes.

```python
# policies/base/policy.py 

@dataclass
class Layer:
    """Dataclass used to describe a layer in the policy object"""

    weight: Any = None
    bias: Any = None
    n_fused: Any = None
    replace: Any = None
    reversed: Any = None
    ignore_checker: bool = False
```

* `weight` and `bias` are the names of the weight and bias tensors, respectively. You can use the syntax such as `[ ]` or `.` to the tensor names.
* `n_fused` is the number of areas used in fused layers. For example, `GPT2` and `TransfoXL` have fused attention layers that consist of query, key and value. The layers should not be simply chunked by the number of GPUs. Instead, they should be divided into the query, key and value areas first.
* `replace` is the layer that you want to replace an existing layer with. The parallelization process by the tensor slicing method involves All-Reduce operations to collect tensors from all GPUs. (you can check it in [this paper](https://arxiv.org/abs/1909.08053)) So, we need to insert some layer like `AllReduceLinear` to replace the existing `nn.Linear` layer.
* `reversed` is used to indicate whether tensors are reversed or not. Some models such as `GPT1` and `GPT2` use the `transformers.modeling_utils.Conv1D` layer instead of the `nn.Linear` layer. These layers store weight and bias tensors reversed. 
* `ignore_checker` is used when you want to ignore errors in case the layers do not exist. Some models like `Bert`, `Roberta` and `Electra` have only encoder layers. but for Huggingface, these models are also designed to be able to used as decoders. In these models, some layers (called `CrossAttention layer`) may or may not be created depending on the configuraions. In this case, you can use `ignore_checker` option to ignore errors even if the layers do not always exist. <---Î≠î ÏÜåÎ¶¨ÏûÑ?


#### 5.1.2. `replace_arguments()`
```python
# example of `replace_arguemens()` method

@staticmethod
def replace_arguments(config, world_size):
    return {
        # 1. reduce hidden size
        "attention.self.embed_dim": config.hidden_size // world_size,
            
        # 2. reduce number of heads
        "attention.self.num_heads": config.num_attention_heads // world_size,
    }
```
The following is a example of `replace_arguments()` method. To parallelize most models, some arguments like number of attention heads and hidden size must be changed. In this case, you can write changes of arguments in the `replace_arguemnts()` method. It will be applied when model start to parallelize.
<br><br>

#### 5.1.3. `replace_modules()`
```python
# example of `replace_modules()` method

@staticmethod
def replace_modules():
    return {
        "BartAttention": BartAttention_,
    }
```
The following is a example of `replace_modules()` method. in some cases, parallelization is impossible due to implementation of Huggingface transformers. So we provide `replace_modules()` method. This allows you to change the codes of exsting layers. You can check more example in the `parallelformers/transformers` directory.
<br><br>

#### 5.1.4. Applying custom policy object
- Finally, input the class of the custom policy as `custom_policies` argument.
```python
from parallelformers import parallelize
from your_codes import YourPolicy

model = Model()
parallelize(model, num_gpus=4, fp16=True, custom_policies=YourPolicy)
```

- You can also input list of multiple polices class if the model requires multiple policies objects.
```python
from parallelformers import parallelize
from your_codes import YourEncoderPolicy, YourDecoderPolicy

model = Model()
parallelize(model, num_gpus=4, fp16=True, custom_policies=[YourEncoderPolicy, YourDecoderPolicy])
```
<br>

### 5.2. Multiple Model Parallelization
```python
# example of multiple model parallelization

parallelize(model_1, num_gpus=4, fp16=True, master_port=29500)
parallelize(model_2, num_gpus=4, fp16=True, master_port=29501)
```
The following is example of multiple model parallelization. It is helpful to change the `master_port` if you want to parallelize multiple models on the same main process. We need to have difference network addresses to create two difference process groups.
<br><br>

### 5.3. Issues about Docker
#### 5.3.1. Issue about multiple model parallelization in Docker containers.
Some limitations are existed if you use `parallelformers` with Docker. In Docker containsers, you can't parallelize multiple models on the same GPUs. For example, If you parallelize first model to GPUs `[0, 1, 2, 3]`, you can not parallelize second model to GPUs `[0, 1, 2, 3]` again. 

This is a bug between Docker and `multiprocessing` package in the Python. There are some bugs when we tried parallelization with the `fork` and `forkserver` method, So we are currently using `spawn` method for multiprocessing. If you already have semophores on your main process and you are using the `multiprocessing` module with `spawn` method, Python try to check leaked semaphores (please check [here](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)). Leaked semaphore checker will not be executed for first model because you don't have any semaphore objects in your main process, but it will be executed for second model. And this leaked semaphore checker makes some problems in the Docker containers. (you can check more details [here](https://stackoverflow.com/questions/54650904/python-multiprocessing-crashes-docker-container), And we are currently working to solve this issue)
<br><br>

#### 5.3.2. Issue about shared memory size
In addition, you need to increase the shared memory size if you want to use distributed backed such as `nccl` in Docker containers (The default is set to 2mb, but it is too small)

<br><br>

## Q&A
#### Why doesn't the GPU usage decrease by exactly _n_ times when I parallelize on _n_ GPUs?

There are three possible reasons. 

1. There are non-parallelizable areas in the model. For example, embedding, normalization and lm head layers can NOT be parallelized, resulting that they are copied to all GPUs.
2. We need to allocate shared memory areas for inter-process communication. Since this shared memory is allocated across all GPU processes, the GPU usage should increase.
3. When input tensors are copied to all GPUs, the GPU usage can increase.

  
#### How many GPUs are good to use when parallelizing a model?
We recommend you keep the number of GPUs as least as possible.

#### Why are some models not supported?
There are several factors.
Models are partly supported or not supported if they ...

1. have dynamically changed layers. We only can parallelize static layers because the parallelization process should be completed before the forward pass. But some models' layers (e.g., `BigBird's Self-Attention`) can change dynamically during the forward pass and ends up to unparallelization. For example, `BigBirdPegasus` contains `BigBird's Self-Attention` in its encoder layers, so they can't be parallelized.

2. have convolutional layers. The convolution operation is not compatible with the tensor slicing method. For example, the attention layers of `ConvBERT` and all the layers of `SqueezeBERT` consist of convolutions, so they can not be parallelized.

It is worth mentioning that although OpenAI's `GPT1` and `GPT2` also use convolutional operations, they can be parallelized because they actually perform matrix multiplication-based operations rather than actual convolutional operations. 
(Check the implementations of the `transformers.modeling_utils.Conv1D` layer)

3. have n-gram attention layers. We conducted several parallelization experiments with `ProphetNet` that adopts the N-gram attention mechanism. Unfortunately, we found the results after the parallelization are not the same as the original representations for some reason.

4. adopt `EncoderDecoderModel`. The `EncoderDecoderModel` framework conflicts with our `AutoPolicy` mechanism. Therefore, when using the `EncoderDecoderModel` framework, you have to write your own custom policy objects.

5. can not be serialized. When transfering a model to other processes, the model's weights must be serialized. Thus, the models that are not serializable such as `RAG` are do not support parallelization.

#### How to contribute?
Refer to [CONTRIBUTING.md](CONTRIBUTING.md).


## Citation
If you find this library useful, please consider citing:

```
@misc{parallelformers,
  author       = {Ko, Hyunwoong},
  title        = {Parallelformers: An Efficient Model Paraelleization Toolkit for Deployment},
  howpublished = {\url{https://github.com/tunib-ai/parallelformers}},
  year         = {2021},
}
```

## LICENSE
`Parallelformers` is licensed under the terms of the Apache License 2.0.

Copyright 2021 TUNiB inc. https://www.tunib.ai. All Rights Reserved.
