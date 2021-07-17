python ../image_model.py \
  --test-name="FP32 & Non-PF"\
  --name='facebook/deit-base-patch16-224' \
  --gpu-from=0 \
  --gpu-to=1

python ../image_model.py \
  --test-name="FP16 & Non-PF"\
  --name='facebook/deit-base-patch16-224' \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ../image_model.py \
  --test-name="FP32 & PF"\
  --name='facebook/deit-base-patch16-224' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ../image_model.py \
  --test-name="FP16 & PF"\
  --name='facebook/deit-base-patch16-224' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
