python ../question_answering.py \
  --test-name="FP32 & Non-PF"\
  --name="albert-base-v2" \
  --gpu-from=0 \
  --gpu-to=1

python ../question_answering.py \
  --test-name="FP16 & Non-PF"\
  --name="albert-base-v2" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ../question_answering.py \
  --test-name="FP32 & PF"\
  --name="albert-base-v2" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ../question_answering.py \
  --test-name="FP16 & PF"\
  --name="albert-base-v2" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
