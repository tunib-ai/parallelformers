python detr_model.py \
  --test-name="FP32 & Non-PF"\
  --name='facebook/detr-resnet-50' \
  --gpu-from=0 \
  --gpu-to=1

python detr_model.py \
  --test-name="FP16 & Non-PF"\
  --name='facebook/detr-resnet-50' \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python detr_model.py \
  --test-name="FP32 & PF"\
  --name='facebook/detr-resnet-50' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python detr_model.py \
  --test-name="FP16 & PF"\
  --name='facebook/detr-resnet-50' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
