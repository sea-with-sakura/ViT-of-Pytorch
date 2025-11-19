export CUDA_VISIBLE_DEVICES=2

python src/train.py \
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
       --train-steps 15000 \
       --lr 0.03 \
       --wd 0.0 \
       --warmup-steps 500 \
