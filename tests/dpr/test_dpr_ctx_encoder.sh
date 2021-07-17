python dpr_ctx_encoder.py \
  --test-name="FP32 & Non-PF"\
  --name="facebook/dpr-ctx_encoder-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1

python dpr_ctx_encoder.py \
  --test-name="FP16 & Non-PF"\
  --name="facebook/dpr-ctx_encoder-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python dpr_ctx_encoder.py \
  --test-name="FP32 & PF"\
  --name="facebook/dpr-ctx_encoder-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python dpr_ctx_encoder.py \
  --test-name="FP16 & PF"\
  --name="facebook/dpr-ctx_encoder-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
