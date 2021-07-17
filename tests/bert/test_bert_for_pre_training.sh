python ../pre_training.py \
  --test-name="FP32 & Non-PF"\
  --name="bert-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1

python ../pre_training.py \
  --test-name="FP16 & Non-PF"\
  --name="bert-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ../pre_training.py \
  --test-name="FP32 & PF"\
  --name="bert-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ../pre_training.py \
  --test-name="FP16 & PF"\
  --name="bert-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
