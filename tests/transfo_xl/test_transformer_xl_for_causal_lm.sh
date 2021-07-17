python ../causal_lm.py \
  --test-name="FP32 & Non-PF"\
  --name="transfo-xl-wt103" \
  --gpu-from=0 \
  --gpu-to=1

python ../causal_lm.py \
  --test-name="FP32 & PF"\
  --name="transfo-xl-wt103" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf
