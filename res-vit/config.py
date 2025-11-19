import argparse
from utils import process_config


def set_model_architecture(model_args, model_arch):
    """
    Set model architecture parameters based on model_arch choice
    
    Args:
        model_args: ModelArgs instance to update
        model_arch: String specifying model architecture ('b16', 'b32', 'l16', 'l32', 'h14')
    
    Returns:
        Updated ModelArgs instance
    """
    if model_arch == 'b16':
        model_args.dim = 768
        model_args.mlp_dim = 3072
        model_args.n_heads = 12
        model_args.n_layers = 12
        model_args.patch_size = (16, 16)
    elif model_arch == 'b32':
        model_args.dim = 768
        model_args.mlp_dim = 3072
        model_args.n_heads = 12
        model_args.n_layers = 12
        model_args.patch_size = (32, 32)
    elif model_arch == 'l16':
        model_args.dim = 1024
        model_args.mlp_dim = 4096
        model_args.n_heads = 16
        model_args.n_layers = 24
        model_args.patch_size = (16, 16)
    elif model_arch == 'l32':
        model_args.dim = 1024
        model_args.mlp_dim = 4096
        model_args.n_heads = 16
        model_args.n_layers = 24
        model_args.patch_size = (32, 32)
    elif model_arch == 'h14':
        model_args.dim = 1280
        model_args.mlp_dim = 5120
        model_args.n_heads = 16
        model_args.n_layers = 32
        model_args.patch_size = (14, 14)
    
    return model_args


def get_num_classes_for_dataset(dataset_name):
    """
    根据数据集名称自动获取类别数量
    
    Args:
        dataset_name (str): 数据集名称
        
    Returns:
        int: 类别数量
    """
    dataset_classes = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'ImageNet': 1000,
        'TinyImageNet': 200
    }
    
    # 返回对应数据集的类别数，如果找不到则返回默认值1000
    return dataset_classes.get(dataset_name, 1000)


def config_to_model_args(ModelArgs,config):
    """
    将配置对象转换为ModelArgs对象
    
    Args:
        config: 配置对象，包含模型训练所需的所有参数
        
    Returns:
        ModelArgs: 配置好的模型参数对象
    """
    # 基本视觉Transformer参数
    model_args = ModelArgs()
    model_args.image_size = (config.image_size, config.image_size)
    model_args.patch_size = (config.patch_size, config.patch_size)
    model_args.n_heads = config.n_heads
    model_args.n_kv_heads = config.n_kv_heads
    model_args.norm_eps = config.norm_eps
    model_args.lora_rank = config.lora_rank
    model_args.dynamic_active_target = config.dynamic_active_target
    model_args.dynamic_start_layer = config.dynamic_start_layer
    model_args.dynamic_router_hdim = config.dynamic_router_hdim
    model_args.dynamic_reserve_initials = config.dynamic_reserve_initials
    model_args.low_rank_dim = config.low_rank_dim
    model_args.block_size = config.block_size
    model_args.use_lora = config.use_lora
    model_args.use_reslr = config.use_reslr
    model_args.num_classes = config.num_classes
    model_args.use_cosine_target_schedule = config.use_cosine_target_schedule
    
    return model_args

def get_eval_config():
    parser = argparse.ArgumentParser("Visual Transformer Evaluation")

    # basic config
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default="../weights/pytorch/imagenet21k+imagenet2012_ViT-B_16-224.pth", help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[224, 384])
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers")
    parser.add_argument("--data-dir", type=str, default='../data/', help='data folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for evaluation", choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--patch-size", type=int, default=16, help="patch size")
    
    # Evaluation hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--device", type=str, default='cuda:2', help="device to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of GPUs to use")

    config = parser.parse_args()

    if config.num_classes == 1000:
        config.num_classes = get_num_classes_for_dataset(config.dataset)

    return config


def get_train_config():
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="reslr", help="experiment name")
    parser.add_argument("--swanlab", default=True, action='store_true', help='flag of turning on swanlab')
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default="../weights/pytorch/imagenet21k+imagenet2012_ViT-B_16-224.pth", help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[224, 384])
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers")
    parser.add_argument("--data-dir", type=str, default='../data/', help='data folder')
    parser.add_argument("--dataset", type=str, default='CIFAR100', help="dataset for fine-tunning/evaluation", 
                                               choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--patch-size", type=int, default=16, help="patch size")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")  # 降低学习率，更适合AdamW
    parser.add_argument("--wd", type=float, default=0.05, help="weight decay")  # 增加权重衰减，更适合ViT和AdamW
    parser.add_argument("--train-steps", type=int, default=15000, help="number of training/fine-tunning steps")
    parser.add_argument("--warmup-steps", type=int, default=500, help="learning rate warm up steps")
    parser.add_argument("--print-freq", type=int, default=100, help="print frequency")
    parser.add_argument("--device", type=str, default='cuda:4', help="device to use for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    # AdamW optimizer hyperparameters
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 parameter")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2 parameter")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon parameter")
    
    # Cosine scheduler hyperparameters
    parser.add_argument("--lr-scheduler", type=str, default="cosine_with_warmup", help="learning rate scheduler", choices=["cosine", "cosine_with_warmup"])
    parser.add_argument("--min-lr", type=float, default=1e-6, help="minimum learning rate")
    
    # Dynamic weight adjustment parameters
    parser.add_argument("--use_lora", type=bool, default=True, help="use LoRA for fine-tuning")
    parser.add_argument("--use_reslr", type=bool, default=True, help="use residual lora for fine-tuning")

    parser.add_argument("--initial-lambda-active", type=float, default=0.01, help="initial lambda_active value")
    parser.add_argument("--initial-lambda-distill", type=float, default=0.1, help="initial lambda_distill value")
    parser.add_argument("--lambda-router-entropy", type=float, default=0.0001, help="weight for router entropy regularization (to prevent routing collapse)")
    
    # Dynamic target scheduling
    parser.add_argument("--use-cosine-target-schedule", type=bool, default=False, 
                        help="use cosine annealing schedule for dynamic_active_target (from 1.0 to target)")

    parser.add_argument("--n_heads", type=int, default=12, help="number of heads")
    parser.add_argument("--n_kv_heads", type=int, default=12, help="number of kv heads")
    parser.add_argument("--norm_eps", type=float, default=1e-5, help="normalization epsilon")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--dynamic_active_target", type=float, default=0.6, 
                        help="final dynamic active target ratio (when use_cosine_target_schedule=True, will decay from 1.0 to this value)")
    parser.add_argument("--dynamic_start_layer", type=int, default=2, help="layer index to start dynamic routing")
    parser.add_argument("--dynamic_router_hdim", type=int, default=512, help="hidden dimension for dynamic router")
    parser.add_argument("--dynamic_reserve_initials", type=int, default=1, help="number of initial layers to reserve")
    parser.add_argument("--low_rank_dim", type=int, default=256, help="low-rank dimension for compression")
    parser.add_argument("--block_size", type=int, default=2, help="block size for grouping (minimum 2)")

    config = parser.parse_args()

    if config.num_classes == 1000:
        config.num_classes = get_num_classes_for_dataset(config.dataset)

    # model config
    process_config(config)
    return config