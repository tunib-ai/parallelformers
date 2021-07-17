python ibert_model.py \
  --test-name="FP32 & Non-PF"\
  --name="kssteven/ibert-roberta-base" \
  --gpu-from=0 \
  --gpu-to=1

python ibert_model.py \
  --test-name="FP16 & Non-PF"\
  --name="kssteven/ibert-roberta-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ibert_model.py \
  --test-name="FP32 & PF"\
  --name="kssteven/ibert-roberta-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ibert_model.py \
  --test-name="FP16 & PF"\
  --name="kssteven/ibert-roberta-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
