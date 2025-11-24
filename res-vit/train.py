import os
import torch
import numpy as np
from model import Transformer, ModelArgs
from config import get_train_config, set_model_architecture,config_to_model_args
from data_loaders import *
from utils import *
from transformers import get_cosine_schedule_with_warmup


def train_epoch(epoch, model, data_loader, optimizer, metrics, config, lr_scheduler=None, lambda_active=10.0, lambda_distill=1.0, lambda_class=10.0, lambda_router_entropy=0.01, device=torch.device('cpu'), total_steps=15000):
    metrics.reset()
    if metrics.writer is not None:
        metrics.writer.set_step(epoch * len(data_loader), mode='train')

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):

        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        # 更新动态激活目标值
        current_step = epoch * len(data_loader) + batch_idx
        if model.use_reslr:
            model.criterion_active.update_target(current_step, total_steps)

        optimizer.zero_grad()

        c_loss, a_loss, d_loss, r_entropy, active_metric = model(batch_data, batch_target)
        
        # 记录每个layer的激活率到swanlab
        if metrics.writer is not None and metrics.writer.enabled:
            layer_activations = {}
            # 假设acts列表存储了每个layer的激活率w
            for i, w in enumerate(model.acts if hasattr(model, 'acts') else []):
                # 计算当前layer的平均激活率
                avg_activation = w.mean().item()
                layer_activations[f'layer_{i}'] = avg_activation
            
            # 使用add_scalars记录所有layer的激活率到一个表中
            if layer_activations:
                metrics.writer.add_scalars('layer_activation_rates', layer_activations)
        
        if model.use_reslr:
            # total_loss = c_loss + lambda_active * a_loss + lambda_distill * d_loss - lambda_router_entropy * r_entropy
            # 注意：负号表示最大化熵，防止塔陷
            total_loss = lambda_class * c_loss + lambda_active * a_loss + lambda_distill * d_loss - lambda_router_entropy * r_entropy
        else:
            total_loss = lambda_class * c_loss
            a_loss = torch.tensor(0.0)
            d_loss = torch.tensor(0.0)
            r_entropy = torch.tensor(0.0)
            active_metric = {
                'non_low_rank_ratio': torch.tensor(0.0),
                'current_target': torch.tensor(0.0)
            }
        
        total_loss.backward()
        if config.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        logits = model.logits
        acc1, acc5 = accuracy(logits, batch_target, topk=(1, 5))

        metrics.writer.set_step(epoch * len(data_loader) + batch_idx, mode='train')
        metrics.update('loss', total_loss.item())
        metrics.update('c_loss', c_loss.item())
        metrics.update('a_loss', a_loss.item())
        metrics.update('d_loss', d_loss.item())
        metrics.update('router_entropy', r_entropy.item() if isinstance(r_entropy, torch.Tensor) else r_entropy)
        metrics.update('acc1', acc1.item())
        metrics.update('acc5', acc5.item())
        # active_metric['non_low_rank_ratio']现在是标量，不需要mean()
        active_ratio_value = active_metric['non_low_rank_ratio']
        if isinstance(active_ratio_value, torch.Tensor):
            active_ratio_value = active_ratio_value.item()
        metrics.update('active_ratio', active_ratio_value)
        metrics.update('lr', optimizer.param_groups[0]['lr'])
        metrics.update('current_target', active_metric['current_target'])

        if batch_idx % config.print_freq == 0:
            active_ratio_value = active_metric['non_low_rank_ratio']
            if isinstance(active_ratio_value, torch.Tensor):
                active_ratio = active_ratio_value.item()
            else:
                active_ratio = active_ratio_value
            current_target = active_metric.get('current_target', 0.0)
            router_entropy = r_entropy.item() if isinstance(r_entropy, torch.Tensor) else r_entropy

            print(
                f"Train Epoch: {epoch:03d} Batch: {batch_idx:05d}/{len(data_loader):05d} Acc@1: {acc1.item():.2f}, Acc@5: {acc5.item():.2f} "
                f"Loss: {total_loss.item():.4f} C_Loss: {c_loss.item():.4f} A_Loss: {a_loss.item():.4f} D_Loss: {d_loss.item():.4f} "
                f"ActiveRatio: {active_ratio:.2f} CurrentTarget: {current_target:.2f} RouterEntropy: {router_entropy:.4f} "
                f"LA: {lambda_active:.1e} LD: {lambda_distill:.1e} LC: {lambda_class:.1e} LE: {lambda_router_entropy:.1e}"
            )

    return metrics.result()


def valid_epoch(epoch, model, data_loader, optimizer, metrics, lambda_active=10.0, lambda_distill=1.0, lambda_class=10.0, lambda_router_entropy=0.01, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    c_losses = []
    a_losses = []
    d_losses = []
    router_entropies = []
    acc1s = []
    acc5s = []
    active_ratios = []
    current_targets = []
    
    # 设置writer模式为'val'
    if metrics.writer is not None:
        metrics.writer.set_step(epoch, mode='valid')
    
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            c_loss, a_loss, d_loss, r_entropy, active_metric = model(batch_data, batch_target)
            
            if model.use_reslr:
                total_loss = lambda_class * c_loss + lambda_active * a_loss + lambda_distill * d_loss - lambda_router_entropy * r_entropy
            else:
                total_loss = lambda_class * c_loss
                a_loss = torch.tensor(0.0)  # 设置为0以便显示
                d_loss = torch.tensor(0.0)  # 设置为0以便显示
                r_entropy = torch.tensor(0.0)
                active_metric = {
                    'non_low_rank_ratio': torch.tensor(0.0),  # 设置为0以便显示
                    'current_target': torch.tensor(0.0)  # 设置为0以便显示
                }

            logits = model.logits
            acc1, acc5 = accuracy(logits, batch_target, topk=(1, 5))

            losses.append(total_loss.item())
            c_losses.append(c_loss.item())
            a_losses.append(a_loss.item())
            d_losses.append(d_loss.item())
            router_entropies.append(r_entropy.item() if isinstance(r_entropy, torch.Tensor) else r_entropy)
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())
            # active_metric['non_low_rank_ratio']现在是标量
            active_ratio_value = active_metric['non_low_rank_ratio']
            if isinstance(active_ratio_value, torch.Tensor):
                active_ratios.append(active_ratio_value.item())
            else:
                active_ratios.append(active_ratio_value)
            # 添加对current_target的监控
            if 'current_target' in active_metric:
                current_target = active_metric['current_target']
                # 如果是tensor，调用item()方法；如果是float，直接使用
                if hasattr(current_target, 'item'):
                    current_targets.append(current_target.item())
                else:
                    current_targets.append(current_target)

    loss = np.mean(losses)
    c_loss = np.mean(c_losses)
    a_loss = np.mean(a_losses)
    d_loss = np.mean(d_losses)
    router_entropy = np.mean(router_entropies) if router_entropies else 0.0
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    active_ratio = np.mean(active_ratios)
    current_target = np.mean(current_targets) if current_targets else 0.0
    
    metrics.update('loss', loss)
    metrics.update('c_loss', c_loss)
    metrics.update('a_loss', a_loss)
    metrics.update('d_loss', d_loss)
    metrics.update('router_entropy', router_entropy)
    metrics.update('acc1', acc1)
    metrics.update('acc5', acc5)
    metrics.update('active_ratio', active_ratio)
    metrics.update('lr', optimizer.param_groups[0]['lr'])
    metrics.update('current_target', current_target)
    
    return metrics.result()


def main():
    # get config
    config = get_train_config()
    device = config.device
    print(f"Using device: {device}")
    set_seed(config.seed if hasattr(config, 'seed') else 42)
    seed = config.seed if hasattr(config, 'seed') else 42

    # swanlab
    writer = SwanLabWriter(config.summary_dir, config.swanlab, config.swanlab_flag)
    print_config(config)

    # metric tracker
    metric_names = ['loss', 'c_loss', 'a_loss', 'd_loss', 'router_entropy', 'acc1', 'acc5', 'active_ratio', 'lr', 'current_target',]
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    model_args = config_to_model_args(ModelArgs, config)
    model_args = set_model_architecture(model_args, config.model_arch)
    model = Transformer(model_args)
    if config.checkpoint_path:
        model = load_pretrained_with_mapping(model, config.checkpoint_path, strict=False, config=config)
        print("Load pretrained weights from {} with mapping".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)

    # use_lora
    if model_args.use_lora:
        save_trainable_weights_info(model, os.path.join(config.summary_dir, "trainable_para.json"))

    # create dataloader
    print("create dataloaders")
    train_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train',
                    seed=seed)
    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val',
                    seed=seed)

    # create optimizers and lr scheduler
    print("create criterion and optimizer")
    epochs = config.train_steps // len(train_dataloader)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        betas=(config.beta1, config.beta2),
        eps=config.eps)

    print("create lr scheduler")
    if config.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=config.min_lr)
    elif config.lr_scheduler == 'cosine_with_warmup':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = config.warmup_steps, 
            num_training_steps = config.train_steps)

    
    # 打印初始GPU内存使用情况
    print_gpu_memory_usage(model, optimizer, device=device, stage="Initial")
    
    # start training
    print("start training")
    best_acc = 0.0
    lambda_active, lambda_distill, lambda_class = config.initial_lambda_active, config.initial_lambda_distill, config.initial_lambda_class
    lambda_router_entropy = config.lambda_router_entropy
    
    
    print(f"Training for {epochs} epochs based on {config.train_steps} steps")
    for epoch in range(epochs):  # 从0开始，到epochs-1结束
        log = {'epoch': epoch}
        log.update({'lambda_active': lambda_active, 'lambda_distill': lambda_distill, 'lambda_class': lambda_class, 'lambda_router_entropy': lambda_router_entropy})

        # train the model
        model.train()
        result = train_epoch(epoch, model, train_dataloader, optimizer, train_metrics, config, 
                            lr_scheduler if config.lr_scheduler == 'cosine_with_warmup' else None,
                            lambda_active, lambda_distill, lambda_class, lambda_router_entropy, device, total_steps=config.train_steps)
        log.update(result)
        
        if config.lr_scheduler == 'cosine':
            lr_scheduler.step()

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, optimizer, valid_metrics, 
                          lambda_active, lambda_distill, lambda_class, lambda_router_entropy, device)
        log.update(**{'val_' + k: v for k, v in result.items()})
        
        # best acc
        best = False
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best = True
        
        # save model
        save_model(config.checkpoint_dir, model, best)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    main()