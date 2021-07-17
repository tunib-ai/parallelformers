python ../pre_training.py \
  --test-name="FP32 & Non-PF"\
  --name="transfo-xl-wt103" \
  --gpu-from=0 \
  --gpu-to=1

python ../pre_training.py \
  --test-name="FP32 & PF"\
  --name="transfo-xl-wt103" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf
