python clip_model.py \
  --test-name="FP32 & Non-PF"\
  --name='openai/clip-vit-base-patch32' \
  --gpu-from=0 \
  --gpu-to=1

python clip_model.py \
  --test-name="FP32 & PF"\
  --name='openai/clip-vit-base-patch32' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf
