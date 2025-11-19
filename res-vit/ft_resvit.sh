export CUDA_VISIBLE_DEVICES=2

# 基础训练配置
python res-vit/train.py \
      --exp-name ft \
      --n-gpu 1 \
      --swanlab  \
      --model-arch b16 \
      --checkpoint-path weights/pytorch/imagenet21k+imagenet2012_ViT-B_16-224.pth \
      --image-size 224 \
      --batch-size 32 \
      --num-workers 1 \
      --data-dir data/ \
      --dataset CIFAR100 \
      --num-classes 100 \
      
      # LoRA和动态路由配置
      --lora_rank 48 \
      --dynamic_active_target 0.4 \
      --dynamic_router_hdim 512 \
      --dynamic_start_layer 1 \
      --dynamic_reserve_initials 2 \
      --lambda_active 10.0 \
      --low_rank_dim 256 \
      --lambda_distill 1.0 \
      --distill_mode \
      --block_size 4 \
      --block_routing_start_epoch 1 \
      
      # 训练超参数
      --batch_size 1 \
      --accum_iter 32 \
      --epochs 20 \
      --warmup_epochs 2 \
      --save_freq 1 \
      --blr 0.00015 \
      --min_lr 5e-7 \
      --weight_decay 0.05 \
      --dropout 0.05

echo "Training completed!"