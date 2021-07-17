python ../image_model.py \
  --test-name="FP32 & Non-PF"\
  --name='google/vit-base-patch16-224-in21k' \
  --gpu-from=0 \
  --gpu-to=1

python ../image_model.py \
  --test-name="FP16 & Non-PF"\
  --name='google/vit-base-patch16-224-in21k' \
  --gpu-from=0 \
  --gpu-to=1 \
  --fp16

python ../image_model.py \
  --test-name="FP32 & PF"\
  --name='google/vit-base-patch16-224-in21k' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf

python ../image_model.py \
  --test-name="FP16 & PF"\
  --name='google/vit-base-patch16-224-in21k' \
  --gpu-from=0 \
  --gpu-to=1 \
  --use-pf \
  --fp16
