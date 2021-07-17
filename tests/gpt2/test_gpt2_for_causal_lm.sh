python ../causal_lm.py \
  --test-name="FP32 & Non-PF"\
  --name="gpt2" \
  --gpu-from=0 \
  --gpu-to=1

python ../causal_lm.py \
  --test-name="FP16 & Non-PF"\
  --name="gpt2" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ../causal_lm.py \
  --test-name="FP32 & PF"\
  --name="gpt2" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ../causal_lm.py \
  --test-name="FP16 & PF"\
  --name="gpt2" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
