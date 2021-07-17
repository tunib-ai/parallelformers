python xlm_prophetnet_model.py \
  --test-name="FP32 & Non-PF"\
  --name="microsoft/xprophetnet-large-wiki100-cased" \
  --gpu-from=0 \
  --gpu-to=1

python xlm_prophetnet_model.py \
  --test-name="FP16 & Non-PF"\
  --name="microsoft/xprophetnet-large-wiki100-cased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python xlm_prophetnet_model.py \
  --test-name="FP32 & PF"\
  --name="microsoft/xprophetnet-large-wiki100-cased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python xlm_prophetnet_model.py \
  --test-name="FP16 & PF"\
  --name="microsoft/xprophetnet-large-wiki100-cased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
