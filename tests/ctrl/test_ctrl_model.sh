python ../model.py \
  --test-name="FP32 & Non-PF"\
  --name="ctrl" \
  --gpu-from=0 \
  --gpu-to=1

python ../model.py \
  --test-name="FP32 & PF"\
  --name="ctrl" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

