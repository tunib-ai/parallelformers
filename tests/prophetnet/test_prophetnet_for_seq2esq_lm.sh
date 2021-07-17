python prophetnet_for_seq2seq_lm.py \
  --test-name="FP32 & Non-PF"\
  --name="microsoft/prophetnet-large-uncased" \
  --gpu-from=0 \
  --gpu-to=1

python prophetnet_for_seq2seq_lm.py \
  --test-name="FP16 & Non-PF"\
  --name="microsoft/prophetnet-large-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python prophetnet_for_seq2seq_lm.py \
  --test-name="FP32 & PF"\
  --name="microsoft/prophetnet-large-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python prophetnet_for_seq2seq_lm.py \
  --test-name="FP16 & PF"\
  --name="microsoft/prophetnet-large-uncased" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
