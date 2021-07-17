python speech_to_text_for_seq2seq_lm.py \
  --test-name="FP32 & Non-PF"\
  --name="facebook/s2t-small-librispeech-asr" \
  --gpu-from=0 \
  --gpu-to=1

python speech_to_text_for_seq2seq_lm.py \
  --test-name="FP32 & PF"\
  --name="facebook/s2t-small-librispeech-asr" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

