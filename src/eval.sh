export CUDA_VISIBLE_DEVICES=2

python src/eval.py \
       --n-gpu 1 \
       --model-arch b16 \
       --checkpoint-path experiments/save/ft_CIFAR10_bs32_lr0.03_wd0.0_251030_134830/checkpoints/best.pth \
       --image-size 224 \
       --batch-size 32 \
       --num-workers 1 \
       --data-dir data/ \
       --dataset CIFAR10 \
       --num-classes 10 \
