## Policy Class
In Parallelformers, every model has its own [`Policy`](https://github.com/tunib-ai/parallelformers/blob/2495f3dbb34b2dded81fa909b650f1fe788cc9ef/parallelformers/policies/base/policy.py#L43) classes that manage the overall parallelization configurations. (Check [this](https://github.com/tunib-ai/parallelformers/tree/docs/parallelformers/policies).) In most cases, you don't have to care about them because the policies of most Hugging Face models are pre-defined in the [`AutoPolicy`](parallelformers/policies/base/auto.py#L192) class. If you want to use a new model that is not in the `AutoPolicy`, you need to add a `Policy` class for yourself. Below are the basic syntax related to the `Policy` class.

### Layer Class
Most methods in the `Policy` class return a list of `Layer` classes. 
For details of arguments of the Layer class, refer to the [docs](https://github.com/tunib-ai/parallelformers/blob/docs/parallelformers/policies/base/policy.py).

### `replace_arguments()`
The following is a example of `replace_arguments()` method. To parallelize most models, some arguments like number of attention heads and hidden size must be changed. In this case, you can write changes of arguments in the `replace_arguemnts()` method. It will be applied when model start to parallelize.
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

### `replace_modules()`
The following is a example of `replace_modules()` method. in some cases, parallelization is impossible due to implementation of Huggingface transformers. So we provide `replace_modules()` method. This allows you to change the codes of exsting layers. You can check more example in the `parallelformers/transformers` directory.

```python
# example of `replace_modules()` method

@staticmethod
def replace_modules():
    return {
        "BartAttention": BartAttention_,
    }
```

### Applying custom policy object
Finally, input the class of the custom policy as `custom_policies` argument.
```python
from parallelformers import parallelize
from your_codes import YourPolicy

model = Model()
parallelize(model, num_gpus=4, fp16=True, custom_policies=YourPolicy)
```

You can also input list of multiple polices class if the model requires multiple policies objects.
```python
from parallelformers import parallelize
from your_codes import YourEncoderPolicy, YourDecoderPolicy

model = Model()
parallelize(model, num_gpus=4, fp16=True, custom_policies=[YourEncoderPolicy, YourDecoderPolicy])
```
