python bert_generation_model.py \
  --test-name="FP32 & Non-PF"\
  --name="google/bert_for_seq_generation_L-24_bbc_encoder" \
  --gpu-from=0 \
  --gpu-to=1

python bert_generation_model.py \
  --test-name="FP16 & Non-PF"\
  --name="google/bert_for_seq_generation_L-24_bbc_encoder" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python bert_generation_model.py \
  --test-name="FP32 & PF"\
  --name="google/bert_for_seq_generation_L-24_bbc_encoder" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python bert_generation_model.py \
  --test-name="FP16 & PF"\
  --name="google/bert_for_seq_generation_L-24_bbc_encoder" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
