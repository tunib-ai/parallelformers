
# FAQ
### Q. Can I use it on Docker?
Yes, but please note the followings.

I recently found out that ALL errors that occur in environments with limited resources such as docker containers are due to **shared memory size**. So, if you want to use larger models with parallelformers in docker containers, **INCREASE the size of shared memory by --shm_size=?gb or REMOVE the limitation of shared memory size by --ipc=host**. the larger shared memory size is required if you want to use larger model.

### Q. Why doesn't the GPU usage decrease by exactly _n_ times when I parallelize on _n_ GPUs?

There are three possible reasons. 

1. There are non-parallelizable areas in the model. For example, embedding, normalization and lm head layers can NOT be parallelized, resulting that they are copied to all GPUs.
2. We need to allocate shared memory areas for inter-process communication. Since this shared memory is allocated across all GPU processes, the GPU usage should increase.
3. When input tensors are copied to all GPUs, the GPU usage can increase.
  
### Q. How many GPUs are good to use when parallelizing a model?
We recommend you keep the number of GPUs as least as possible.

### Q. Why are some models not supported?
There are several factors.
Models are partly supported or not supported if they ...

1. have dynamically changed layers. 

We only can parallelize static layers because the parallelization process should be completed before the forward pass. But some models' layers (e.g., `BigBird's Self-Attention`) can change dynamically during the forward pass and ends up to unparallelization. For example, `BigBirdPegasus` contains `BigBird's Self-Attention` in its encoder layers, so they can't be parallelized.

2. have convolutional layers. 

The convolution operation is not compatible with the tensor slicing method. For example, the attention layers of `ConvBERT` and all the layers of `SqueezeBERT` consist of convolutions, so they can not be parallelized. It is worth mentioning that although OpenAI's `GPT1` and `GPT2` also use convolutional operations, they can be parallelized because they actually perform matrix multiplication-based operations rather than actual convolutional operations. (Check the implementations of the `transformers.modeling_utils.Conv1D` layer)

3. have n-gram attention layers. 

We conducted several parallelization experiments with `ProphetNet` that adopts the N-gram attention mechanism. Unfortunately, we found the results after the parallelization are not the same as the original representations for some reason.

4. adopt `EncoderDecoderModel`. 

The `EncoderDecoderModel` framework conflicts with our `AutoPolicy` mechanism. Therefore, when using the `EncoderDecoderModel` framework, you have to write your own custom policy objects.

5. can not be serialized.

When transfering a model to other processes, the model's weights must be serialized. Thus, the models that are not serializable such as `RAG` are do not support parallelization.

### Q. Can I parallelize multiple models on same GPUs?
Yes. The following is example of multiple model parallelization. Note it is helpful to change the `master_port` if you want to parallelize multiple models on the same main process.

```python
# example of multiple model parallelization

parallelize(model_1, num_gpus=4, fp16=True, master_port=29500)
parallelize(model_2, num_gpus=4, fp16=True, master_port=29501)
```


