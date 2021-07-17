python dpr_reader.py \
  --test-name="FP32 & Non-PF"\
  --name="facebook/dpr-reader-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1

python dpr_reader.py \
  --test-name="FP16 & Non-PF"\
  --name="facebook/dpr-reader-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python dpr_reader.py \
  --test-name="FP32 & PF"\
  --name="facebook/dpr-reader-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python dpr_reader.py \
  --test-name="FP16 & PF"\
  --name="facebook/dpr-reader-single-nq-base" \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
