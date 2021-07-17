python ../sequence_classification.py \
  --test-name="FP32 & Non-PF"\
  --name="microsoft/deberta-v2-xlarge-mnli" \
  --gpu-from=0 \
  --gpu-to=1
#
python ../sequence_classification.py \
  --test-name="FP16 & Non-PF"\
  --name="microsoft/deberta-v2-xlarge-mnli" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ../sequence_classification.py \
  --test-name="FP32 & PF"\
  --name="microsoft/deberta-v2-xlarge-mnli" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ../sequence_classification.py \
  --test-name="FP16 & PF"\
  --name="microsoft/deberta-v2-xlarge-mnli" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
