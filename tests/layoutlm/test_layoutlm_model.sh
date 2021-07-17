python layoutlm_model.py \
  --test-name="FP32 & Non-PF"\
  --name="microsoft/layoutlm-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1

python layoutlm_model.py \
  --test-name="FP16 & Non-PF"\
  --name="microsoft/layoutlm-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python layoutlm_model.py \
  --test-name="FP32 & PF"\
  --name="microsoft/layoutlm-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python layoutlm_model.py \
  --test-name="FP16 & PF"\
  --name="microsoft/layoutlm-base-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
