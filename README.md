# Vision Transformer Implementations in PyTorch

This repository contains two implementations of Vision Transformer (ViT) models in PyTorch:

1. **Standard Vision Transformer** (`src/`): A clean implementation of the original ViT architecture
2. **Residual Vision Transformer (Res-ViT)** (`res-vit/`): An enhanced ViT implementation with advanced features like dynamic activation and residual learning

## Project Structure

```
vision-transformer-pytorch/
├── src/                    # Standard Vision Transformer implementation
│   ├── model.py            # ViT model definition
│   ├── config.py           # Configuration settings
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── data_loaders.py     # Data loading utilities
│   └── utils.py            # Helper functions
├── res-vit/                # Residual Vision Transformer implementation
│   ├── model.py            # Enhanced Res-ViT model with dynamic features
│   ├── config.py           # Configuration for Res-ViT
│   ├── train.py            # Training script with advanced loss functions
│   ├── eval.py             # Evaluation script
│   └── model_utils.py      # Additional utilities for Res-ViT
├── weights/                # Directory for model weights
└── requirements.txt        # Python dependencies
```

## Standard Vision Transformer (src/)

The standard implementation follows the original Vision Transformer architecture with the following components:

- **PositionEmbedding**: Learnable positional embeddings for image patches
- **SelfAttention**: Multi-head self-attention mechanism
- **MLPBlock**: Feed-forward network with GELU activation
- **EncoderBlock**: Transformer encoder block combining attention and MLP
- **VisionTransformer**: Main model class with patch embedding, transformer encoder, and classification head

### Supported Model Configurations

- **ViT-B/16**: Base model with 16×16 patch size
- **ViT-B/32**: Base model with 32×32 patch size
- **ViT-L/16**: Large model with 16×16 patch size
- **ViT-L/32**: Large model with 32×32 patch size
- **ViT-H/14**: Huge model with 14×14 patch size

## Residual Vision Transformer (res-vit/)

The Res-ViT implementation extends the standard ViT with several advanced features:

- **Dynamic Activation**: Router modules that dynamically determine which tokens/layers to process
- **LoRA Support**: Low-Rank Adaptation for efficient fine-tuning
- **Multi-Component Loss**: Combined classification, activation, and distillation losses
- **Active Learning Mechanism**: Maintains target activation ratios during training
- **Residual Learning**: Enhanced residual connections

### Key Components

- **RouterModule**: Determines active/inactive tokens dynamically
- **ActiveLoss**: Controls the activation ratio of tokens
- **DistillLoss**: Knowledge distillation between active components
- **LoRAModule**: Implements Low-Rank Adaptation for parameter efficiency

## Training

### Standard ViT Training

```bash
cd src
python train.py --dataset CIFAR100 --model-arch b16 --batch-size 32 --lr 0.03
```

### Res-ViT Training

```bash
cd res-vit
python train.py --dataset CIFAR100 --model-arch b16 --use-reslr True --dynamic-active-target 0.4
```

## Evaluation

### Standard ViT Evaluation

```bash
cd src
python eval.py --checkpoint-path /path/to/checkpoint.pth --dataset CIFAR100
```

### Res-ViT Evaluation

```bash
cd res-vit
python eval.py --checkpoint-path /path/to/checkpoint.pth --dataset CIFAR100
```

## Features Comparison

| Feature | Standard ViT (src/) | Residual ViT (res-vit/) |
|---------|-------------------|------------------------|
| Basic Transformer Architecture | ✓ | ✓ |
| Multiple Model Sizes | ✓ | ✓ |
| Dynamic Token Activation | ✗ | ✓ |
| LoRA Support | ✗ | ✓ |
| Multi-Component Loss | ✗ | ✓ |
| Distillation | ✗ | ✓ |
| Router Modules | ✗ | ✓ |
| Cosine Target Scheduling | ✗ | ✓ |

## Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Fine-tuning Scripts

The repository includes pre-configured fine-tuning scripts for common datasets:

- **src/FT_CIFAR10.sh**: Fine-tune standard ViT on CIFAR-10
- **src/FT_CIFAR100.sh**: Fine-tune standard ViT on CIFAR-100
- **res-vit/ft_resvit.sh**: Fine-tune Res-ViT on various datasets

## Acknowledgements

This implementation is based on the original Vision Transformer paper:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.