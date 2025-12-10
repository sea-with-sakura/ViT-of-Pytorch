import os
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import gc
from PIL import Image

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res

def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_bs{}_lr{}_wd{}'.format(config.dataset, config.batch_size, config.lr, config.wd)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', 'tb', exp_name)
    config.checkpoint_dir = os.path.join('experiments', 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join('experiments', 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))

    return config


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average']) #type:ignore
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

class SwanLabWriter:
    def __init__(self, log_dir, enabled=True, project_name=None):
        self.enabled = enabled
        self.step = 0
        self.mode = ''
        self.timer = datetime.now()
        if not enabled:
            self.swanlab = None
            return
        else:
            import swanlab
            self.swanlab = swanlab
            self.swanlab.init(project=project_name, config={"log_dir": log_dir})

    def set_step(self, step, mode='train'):
        """设置当前步骤和模式"""
        self.mode = mode
        self.step = step
        
        if step > 0:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
        self.timer = datetime.now()

    def _format_tag(self, tag):
        """格式化标签，添加模式前缀"""
        return f"{tag}/{self.mode}" if self.mode else tag

    def add_scalar(self, tag, data, *args, **kwargs):
        """记录标量值"""
        if self.enabled and self.swanlab:
            self.swanlab.log({self._format_tag(tag): data}, step=self.step)

    def add_scalars(self, tag, data, *args, **kwargs):
        """记录多个标量值"""
        if not (self.enabled and self.swanlab):
            return
            
        if self.mode:
            log_data = {f"{tag}/{k}/{self.mode}": v for k, v in data.items()}
        else:
            log_data = data
        self.swanlab.log(log_data, step=self.step)
        
    def finish(self):
        """结束日志记录"""
        if self.enabled and self.swanlab:
            self.swanlab.finish()

def load_checkpoint(path):
    """ Load weights from a given checkpoint path in npz/pth """
    if path.endswith('pth'):
        state_dict = torch.load(path)['state_dict']
    else:
        raise ValueError("checkpoint format {} not supported yet!".format(path.split('.')[-1]))

    return state_dict

def save_model(save_dir,model, best=False):
    model_filename = str(save_dir + 'current_model.pth')
    torch.save(model, model_filename)

    if best:
        best_model_filename = str(save_dir + 'best_model.pth')
        torch.save(model, best_model_filename)


def load_pretrained_with_mapping(model, pretrained_path, strict, config):
    """
    加载预训练模型权重到新模型，使用预定义的权重名称映射规则
    
    Args:
        model: 要加载权重的目标模型
        pretrained_path: 预训练模型权重的路径
        strict: 是否严格匹配权重名称，默认为False
    
    Returns:
        model: 加载权重后的模型
        missing_keys: 目标模型中未匹配的权重键列表
        unmatched_keys: 预训练模型中未匹配的权重键列表
    """
    # 打印模型结构信息并保存到JSON
    import json
    model_info = {}
    for name, param in model.named_parameters():
        model_info[name] = {
            "shape": list(param.shape),
            "requires_grad": param.requires_grad
        }
    
    # 保存模型信息到JSON文件
    with open(os.path.join(config.summary_dir, "model_structure.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    # 加载预训练模型权重
    if pretrained_path.endswith('pth'):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint
    else:
        raise ValueError(f"不支持的预训练模型格式: {pretrained_path.split('.')[-1]}")
    
    # 打印预训练模型state_dict信息并保存到JSON
    state_dict_info = {}
    for key, tensor in pretrained_state_dict.items():
        state_dict_info[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype)
        }
    
    # 保存state_dict信息到JSON文件
    with open(os.path.join(config.summary_dir, "pretrained_state_dict.json"), "w", encoding="utf-8") as f:
        json.dump(state_dict_info, f, indent=2, ensure_ascii=False)
    
    # 获取目标模型的state_dict
    model_state_dict = model.state_dict()
    
    # 创建新的state_dict，只包含目标模型中存在的权重
    new_state_dict = {}
    missing_keys = []
    unmatched_keys = []
    
    # 用于记录映射过程的字典
    mapping_log = {
        "successful_mappings": [],
        "successful_reshapes": [],
        "unmatched_keys": [],
        "missing_keys": [],
        "summary": {}
    }
    
    # 首先检查目标模型中有哪些权重
    target_keys = set(model_state_dict.keys())
    
    # 预定义的权重名称映射规则
    def map_weight_name(pretrained_key):
        # 处理transformer相关的权重
        if pretrained_key.startswith('transformer.'):
            # 处理编码器层
            if 'encoder_layers.' in pretrained_key:
                # 将transformer.encoder_layers.X替换为layers.X
                new_key = pretrained_key.replace('transformer.encoder_layers.', 'layers.')
                
                # 处理注意力权重
                if '.attn.query' in new_key:
                    return new_key.replace('.attn.query', '.attention.wq')
                elif '.attn.key' in new_key:
                    return new_key.replace('.attn.key', '.attention.wk')
                elif '.attn.value' in new_key:
                    return new_key.replace('.attn.value', '.attention.wv')
                elif '.attn.out' in new_key:
                    return new_key.replace('.attn.out', '.attention.wo')
                
                # 处理MLP权重
                elif '.mlp.fc1' in new_key:
                    return new_key.replace('.mlp.fc1', '.feed_forward.fc1')
                elif '.mlp.fc2' in new_key:
                    return new_key.replace('.mlp.fc2', '.feed_forward.fc2')
                
                # 处理归一化层权重
                elif '.norm1' in new_key:
                    return new_key.replace('.norm1', '.attention_norm.layer_norm')
                elif '.norm2' in new_key:
                    return new_key.replace('.norm2', '.ffn_norm.layer_norm')
                
                return new_key
            
            # 处理最终的归一化层
            elif pretrained_key == 'transformer.norm.bias':
                return 'norm.layer_norm.bias'
            elif pretrained_key == 'transformer.norm.weight':
                return 'norm.layer_norm.weight'
            
            # 处理位置编码 
            elif pretrained_key == 'transformer.pos_embedding.pos_embedding':
                return 'pos_embedding.pos_embedding'
        
        elif pretrained_key == 'embedding.bias' or pretrained_key == 'embedding.weight':
            return pretrained_key
        
        elif pretrained_key == 'cls_token' :
            return pretrained_key

        # 其他情况返回None，表示不映射
        return None
    
    # 处理qkv权重心塑的函数
    def reshape_qkv_weight(pretrained_weight, target_shape, weight_type):
        """
        重塑qkv权重以匹配目标模型
        
        Args:
            pretrained_weight: 预训练模型的权重
            target_shape: 目标模型的权重形状
            weight_type: 权重类型 ('query', 'key', 'value', 'out')
        
        Returns:
            重塑后的权重
        """
        # 预训练模型中qkv权重的形状可能是 [n_heads, head_dim] 或 [dim, n_heads, head_dim]
        # 目标模型中权重的形状可能是 [dim, dim] 或 [dim]
        
        if weight_type in ['query', 'key', 'value']:
            # 对于q, k, v权重，预训练模型可能是 [dim, n_heads, head_dim] 或 [n_heads, head_dim]
            # 目标模型是 [dim, dim] 或 [dim, dim/n_heads]
            
            if len(pretrained_weight.shape) == 3:  # [dim, n_heads, head_dim]
                # 直接重塑为 [dim, dim]
                dim, n_heads, head_dim = pretrained_weight.shape
                return pretrained_weight.reshape(dim, n_heads * head_dim).transpose(0, 1)
            
            elif len(pretrained_weight.shape) == 2:  # [n_heads, head_dim]
                # 重塑为 [n_heads * head_dim]
                n_heads, head_dim = pretrained_weight.shape
                return pretrained_weight.reshape(n_heads * head_dim)


        elif weight_type == 'out':
            # 对于输出权重，预训练模型可能是 [n_heads, head_dim, dim]
            # 目标模型是 [dim, dim]
            
            if len(pretrained_weight.shape) == 3:  # [n_heads, head_dim, dim]
                n_heads, head_dim, dim = pretrained_weight.shape
                return pretrained_weight.reshape(n_heads * head_dim, dim).transpose(0, 1)
            
            elif len(pretrained_weight.shape) == 2:  # [n_heads, head_dim]
                # 重塑为 [n_heads * head_dim]
                n_heads, head_dim = pretrained_weight.shape
                return pretrained_weight.reshape(n_heads * head_dim)
        
        # 如果无法重塑，返回原始权重
        return pretrained_weight
    
    # 应用权重映射
    for pretrained_key, weight in pretrained_state_dict.items():
        # 应用映射规则
        mapped_key = map_weight_name(pretrained_key)
        
        # 如果映射后的键不为None且在目标模型中
        if mapped_key is not None and mapped_key in target_keys:
            # 检查权重形状是否匹配
            if weight.shape == model_state_dict[mapped_key].shape:
                # 确保权重转换为bf16类型，与模型权重类型保持一致
                new_state_dict[mapped_key] = weight
                mapping_log["successful_mappings"].append({
                    "pretrained_key": pretrained_key,
                    "mapped_key": mapped_key,
                    "shape": list(weight.shape),
                    "status": "direct_match"
                })
            else:
                # 检查是否是qkv权重，需要重塑
                weight_type = None
                if '.attention.wq' in mapped_key:
                    weight_type = 'query'
                elif '.attention.wk' in mapped_key:
                    weight_type = 'key'
                elif '.attention.wv' in mapped_key:
                    weight_type = 'value'
                elif '.attention.wo' in mapped_key:
                    weight_type = 'out'
                
                if weight_type is not None:
                    # 尝试重塑权重
                    reshaped_weight = reshape_qkv_weight(weight, model_state_dict[mapped_key].shape, weight_type)
                    if reshaped_weight.shape == model_state_dict[mapped_key].shape:
                        # 确保重塑后的权重转换为bf16类型，与模型权重类型保持一致
                        new_state_dict[mapped_key] = reshaped_weight
                        mapping_log["successful_reshapes"].append({
                            "pretrained_key": pretrained_key,
                            "mapped_key": mapped_key,
                            "original_shape": list(weight.shape),
                            "reshaped_shape": list(reshaped_weight.shape),
                            "target_shape": list(model_state_dict[mapped_key].shape),
                            "weight_type": weight_type
                        })
                    else:
                        mapping_log["unmatched_keys"].append({
                            "pretrained_key": pretrained_key,
                            "mapped_key": mapped_key,
                            "original_shape": list(weight.shape),
                            "reshaped_shape": list(reshaped_weight.shape),
                            "target_shape": list(model_state_dict[mapped_key].shape),
                            "weight_type": weight_type,
                            "reason": "reshape_failed"
                        })
                        if strict:
                            missing_keys.append(mapped_key)
                        else:
                            unmatched_keys.append(pretrained_key)
                else:
                    mapping_log["unmatched_keys"].append({
                        "pretrained_key": pretrained_key,
                        "mapped_key": mapped_key,
                        "original_shape": list(weight.shape),
                        "target_shape": list(model_state_dict[mapped_key].shape),
                        "reason": "shape_mismatch"
                    })
                    if strict:
                        missing_keys.append(mapped_key)
                    else:
                        unmatched_keys.append(pretrained_key)
        elif mapped_key is None:
            # 跳过的权重
            mapping_log["unmatched_keys"].append({
                "pretrained_key": pretrained_key,
                "reason": "no_mapping_rule"
            })
            unmatched_keys.append(pretrained_key)
        else:
            # 映射后但不在目标模型中的权重
            mapping_log["unmatched_keys"].append({
                "pretrained_key": pretrained_key,
                "mapped_key": mapped_key,
                "reason": "mapped_key_not_in_target_model"
            })
            if strict:
                missing_keys.append(mapped_key)
            else:
                unmatched_keys.append(pretrained_key)
    
    # 找出目标模型中未加载的权重
    loaded_keys = set(new_state_dict.keys())
    for key in target_keys:
        if key not in loaded_keys:
            missing_keys.append(key)
            mapping_log["missing_keys"].append({
                "key": key,
                "shape": list(model_state_dict[key].shape),
                "reason": "not_in_pretrained_model"
            })
    
    # 加载权重到模型
    model.load_state_dict(new_state_dict, strict=False)
    
    # 添加汇总信息
    mapping_log["summary"] = {
        "total_pretrained_weights": len(pretrained_state_dict),
        "total_target_weights": len(target_keys),
        "successful_mappings": len(mapping_log["successful_mappings"]),
        "successful_reshapes": len(mapping_log["successful_reshapes"]),
        "missing_keys": len(missing_keys),
        "unmatched_keys": len(unmatched_keys),
        "total_loaded_weights": len(new_state_dict)
    }
    
    # 保存映射日志到JSON文件
    with open(os.path.join(config.summary_dir, "weight_mapping_log.json"), "w", encoding="utf-8") as f:
        json.dump(mapping_log, f, indent=2, ensure_ascii=False)
    
    return model

def save_trainable_weights_info(model, jsonname):
    """
    检查并保存可训练的权重信息到JSON文件
    """
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append({
                'name': name,
                'shape': list(param.shape),
                'num_elements': param.numel()
            })
        else:
            frozen_params.append({
                'name': name,
                'shape': list(param.shape),
                'num_elements': param.numel()
            })
    
    # 计算总参数数量
    total_trainable = sum(p['num_elements'] for p in trainable_params)
    total_frozen = sum(p['num_elements'] for p in frozen_params)
    total_params = total_trainable + total_frozen
    
    # 保存到JSON文件
    import json
    trainable_info = {
        'model_type': 'Vision Transformer with LoRA',
        'total_parameters': total_params,
        'trainable_parameters': total_trainable,
        'frozen_parameters': total_frozen,
        'trainable_ratio': total_trainable / total_params if total_params > 0 else 0,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params
    }
    
    # 保存到当前目录
    with open(jsonname, 'w') as f:
        json.dump(trainable_info, f, indent=2)

def print_gpu_memory_usage(model, optimizer, batch_data=None, batch_target=None, device=None, stage=""):
    """
    打印GPU内存使用情况，包括模型、数据、优化器和梯度的内存占用
    
    Args:
        model: 训练模型
        optimizer: 优化器
        batch_data: 输入数据 (可选)
        batch_target: 目标数据 (可选)
        device: 设备类型
        stage: 当前训练阶段标识
    """
    
    # 获取GPU总内存和已使用内存
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    
    # 计算模型参数内存占用
    model_memory = 0
    for param in model.parameters():
        if param.is_cuda:
            model_memory += param.nelement() * param.element_size()
    
    # 计算模型梯度内存占用
    gradient_memory = 0
    for param in model.parameters():
        if param.is_cuda and param.grad is not None:
            gradient_memory += param.grad.nelement() * param.grad.element_size()
    
    # 计算优化器状态内存占用
    optimizer_memory = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.is_cuda:
                # 参数本身已经在model_memory中计算，这里只计算优化器状态
                if param in optimizer.state:
                    for state in optimizer.state[param].values():
                        if isinstance(state, torch.Tensor):
                            optimizer_memory += state.nelement() * state.element_size()
    
    # 计算数据内存占用
    data_memory = 0
    if batch_data is not None and batch_data.is_cuda:
        data_memory += batch_data.nelement() * batch_data.element_size()
    if batch_target is not None and batch_target.is_cuda:
        data_memory += batch_target.nelement() * batch_target.element_size()
    
    # 计算其他内存占用（包括中间激活、缓存等）
    other_memory = allocated_memory - model_memory - gradient_memory - optimizer_memory - data_memory
    
    # 转换为GB
    def to_gb(bytes_val):
        return bytes_val / (1024 ** 3)
    
    # 打印内存使用情况
    print("\n" + "="*80)
    print(f"GPU Memory Usage Breakdown - {stage}:")
    print(f"Total GPU Memory:     {to_gb(total_memory):.2f} GB")
    print(f"Allocated Memory:     {to_gb(allocated_memory):.2f} GB ({allocated_memory/total_memory*100:.1f}%)")
    print(f"Reserved Memory:      {to_gb(reserved_memory):.2f} GB ({reserved_memory/total_memory*100:.1f}%)")
    print(f"Memory Fragmentation: {to_gb(reserved_memory - allocated_memory):.2f} GB")
    print("-"*80)
    print(f"Model Parameters:     {to_gb(model_memory):.2f} GB ({model_memory/allocated_memory*100:.1f}% of allocated)")
    print(f"Gradients:            {to_gb(gradient_memory):.2f} GB ({gradient_memory/allocated_memory*100:.1f}% of allocated)")
    print(f"Optimizer States:     {to_gb(optimizer_memory):.2f} GB ({optimizer_memory/allocated_memory*100:.1f}% of allocated)")
    print(f"Batch Data:           {to_gb(data_memory):.2f} GB ({data_memory/allocated_memory*100:.1f}% of allocated)")
    print(f"Other (Activations):  {to_gb(other_memory):.2f} GB ({other_memory/allocated_memory*100:.1f}% of allocated)")
    print("="*80 + "\n")


def optimize_memory_usage(device=None):
    """
    优化GPU内存使用
    
    Args:
        device: 设备类型
    """
    
    # 清理Python垃圾回收
    gc.collect()
    
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    
    # 同步CUDA操作
    torch.cuda.synchronize()


def print_config(config):
    message = ''
    message += '----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


def save_routing_visualization(epoch, batch_data, routing_maps, config, mode='train', patch_size=14):
    """
    保存路由可视化图像
    
    Args:
        epoch: 当前epoch编号
        batch_data: 原始输入图像batch [batch, channels, height, width]
        routing_maps: 字典，包含每个block-head的路由值 {block_id: [batch, seq_len, block_size]}
        config: 配置对象，包含summary_dir等路径信息
        mode: 'train' 或 'val'，用于区分训练和验证的可视化
        patch_size: patch的边长，默认14（对应14x14=196个tokens）
    """
    # 创建mode/epoch子文件夹
    epoch_dir = os.path.join(config.summary_dir, mode, f'epoch-{epoch}')
    ensure_dir(epoch_dir)
    
    # 获取第0张原始图片并保存
    original_img = batch_data[0]  # [channels, height, width]
    
    # 反归一化（假设使用ImageNet标准化）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
    original_img = original_img * std + mean
    original_img = torch.clamp(original_img, 0, 1)
    
    # 转换为PIL图像并保存
    original_img_np = (original_img.cpu().numpy() * 255).astype(np.uint8)
    original_img_np = np.transpose(original_img_np, (1, 2, 0))  # [H, W, C]
    original_pil = Image.fromarray(original_img_np)
    original_pil.save(os.path.join(epoch_dir, 'original_image.png'))
    
    # 为每个block-head生成路由可视化图
    for block_id, routing in routing_maps.items():
        # routing形状: [batch, seq_len, block_size]
        # 取第0个样本
        routing_sample = routing[0]  # [seq_len, block_size]
        
        # 对于每个block_size位置，生成一张图
        block_size = routing_sample.shape[-1]
        for pos in range(block_size):
            # 获取当前位置的路由决策
            routing_pos = routing_sample[:, pos]  # [seq_len]
            
            # 排除CLS token (第0个token)
            routing_pos = routing_pos[1:]  # [196] 只取patch tokens
            
            # 将196个token还原到14x14网格
            routing_grid = routing_pos.view(patch_size, patch_size).cpu().numpy()
            
            # 创建可视化图像
            # 在原图上叠加路由决策
            viz_img = original_img_np.copy().astype(np.float32)
            
            # 计算每个patch对应的像素区域
            h, w = original_img_np.shape[:2]
            patch_h = h // patch_size
            patch_w = w // patch_size
            
            # 创建overlay层
            overlay = np.zeros_like(viz_img)
            alpha_mask = np.zeros((h, w), dtype=np.float32)
            
            for i in range(patch_size):
                for j in range(patch_size):
                    y_start = i * patch_h
                    y_end = (i + 1) * patch_h if i < patch_size - 1 else h
                    x_start = j * patch_w
                    x_end = (j + 1) * patch_w if j < patch_size - 1 else w
                    
                    route_value = routing_grid[i, j]
                    

                    if route_value == 0:  # 低秩路径 - 绿色
                        overlay[y_start:y_end, x_start:x_end] = [0, 255, 0]
                        alpha_mask[y_start:y_end, x_start:x_end] = 0.3
                    else:  # 完整路径 - 灰色
                        overlay[y_start:y_end, x_start:x_end] = [128, 128, 128]
                        alpha_mask[y_start:y_end, x_start:x_end] = 0.5
            
            # 混合原图和overlay
            alpha_mask = alpha_mask[:, :, np.newaxis]
            viz_img = (1 - alpha_mask) * viz_img + alpha_mask * overlay
            viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
            
            # 保存可视化图像
            # 计算实际的layer_id
            layer_id = config.dynamic_start_layer + block_id * config.block_size + pos
            viz_pil = Image.fromarray(viz_img)
            viz_pil.save(os.path.join(epoch_dir, f'routing_layer_{layer_id:02d}_block_{block_id}_pos_{pos}.png'))
    
    print(f"Saved {mode} routing visualization for epoch {epoch} to {epoch_dir}")