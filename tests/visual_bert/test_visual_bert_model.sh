python visual_bert_model.py \
  --test-name="FP32 & Non-PF"\
  --name="uclanlp/visualbert-vqa-coco-pre" \
  --gpu-from=0 \
  --gpu-to=1

python visual_bert_model.py \
  --test-name="FP16 & Non-PF"\
  --name="uclanlp/visualbert-vqa-coco-pre" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python visual_bert_model.py \
  --test-name="FP32 & PF"\
  --name="uclanlp/visualbert-vqa-coco-pre" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python visual_bert_model.py \
  --test-name="FP16 & PF"\
  --name="uclanlp/visualbert-vqa-coco-pre" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
