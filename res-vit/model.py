import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from einops import rearrange
from model_utils import repeat_kv, get_indices_from_LRA_mask

@dataclass
class ModelArgs:
    dim: int = 768
    mlp_dim: int = 3072
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = 12
    norm_eps: float = 1e-5  
    lora_rank: int = 8
    dynamic_active_target: float = 0.4
    dynamic_start_layer: int = 2
    dynamic_router_hdim: int = 512
    dynamic_reserve_initials: int = 1
    low_rank_dim: int = 256
    block_size: int = 2
    use_lora: bool = False
    use_reslr: bool = False

    # from config.py
    image_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (16, 16)
    num_classes: int = 100
    dropout: float = 0.15
    num_patches: int = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    device: str = 'cuda'


class DistillLoss(nn.Module):
    """
    蒸馏损失：计算 student 和 teacher 的 cls token 的 MSE 损失
    """
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor):
        """
        Args:
            student_cls: student 路径的 cls token [batch, dim]
            teacher_cls: teacher 路径的 cls token [batch, dim]
        Returns:
            MSE loss between student and teacher cls tokens
        """
        # teacher 作为目标，需要 detach 避免梯度反传
        target = teacher_cls.detach()
        loss = self.criterion(student_cls, target)
        return loss

class ActiveLoss(nn.Module):
    def __init__(self, target, reserve_initials):

        super().__init__()
        self.target = target
        self.reserve_initials = reserve_initials

    @torch.no_grad()
    def metric(self, activation: torch.Tensor):

        activation = activation[:, self.reserve_initials:, :]
        metrics = {}
        non_low_rank_ratio = activation.mean()  # 全局平均
        metrics.update({'non_low_rank_ratio': non_low_rank_ratio})
        metrics.update({'current_target': self.target})
        return metrics

    def forward(self, activation: torch.Tensor):

        activation = activation[:, self.reserve_initials:, :]  # [batch, seq_len, n_layers]
        global_active_ratio = activation.mean()
        target = torch.tensor(self.target, device=activation.device)
        loss = F.mse_loss(global_active_ratio, target)
        
        return loss

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

    def forward(self, x):
        if x.shape[1] != self.pos_embedding.shape[1]:
            min_len = min(x.shape[1], self.pos_embedding.shape[1])
            out = x[:, :min_len] + self.pos_embedding[:, :min_len]
            if x.shape[1] > self.pos_embedding.shape[1]:
                out = torch.cat([out, x[:, min_len:]], dim=1)
        else:
            out = x + self.pos_embedding

        return out


class LoRAModule(nn.Module):
    def __init__(self, in_dim: int, rank: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.out_dim = out_dim
        self.lora_A = nn.Linear(self.in_dim, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_dim, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.01)

    def forward(self, x):
        lora_out = self.lora_B(self.lora_A(x))
        return lora_out

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_lora: bool = False):

        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        
        if use_lora:
            for param in self.layer_norm.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.layer_norm(x)
    

class RouterModule(nn.Module):
    """
    DynamicViT router
    """
    def __init__(
        self, 
        in_dim: int,
        hidden_dim: int,
        reserve_initials: int,
        norm_eps: float,
        block_size: int = 1,
        use_lora: bool = False
    ):
        super().__init__()
        self.block_size = block_size
        self.reserve_initials = reserve_initials
        
        # DynamicViT  Local-Global 
        self.in_conv = nn.Sequential(
            LayerNorm(in_dim, norm_eps, use_lora=use_lora),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, block_size * 2),
        )
        
        nn.init.normal_(self.out_conv[-1].weight, mean=0, std=0.01)
        for i in range(block_size):
            self.out_conv[-1].bias.data[i*2] = 0.0       # pass path
            self.out_conv[-1].bias.data[i*2+1] = 5.0     # keep path (Full Transformer)

    def _router2indices(self, x):
        n = x.shape[-1]
        weights = torch.stack([torch.tensor(2.0 ** (n - 1 - i), dtype=x.dtype, device=x.device) for i in range(n)]).unsqueeze(-1)
        merged = torch.matmul(x.float(), weights.float())
        return merged

    def forward(self, x):
        # x shape: [Batch, SeqLen, Dim]
        B, N, C = x.shape
        x_embed = self.in_conv(x)  # [B, N, hidden_dim]
        
        if self.reserve_initials > 0:
            patch_tokens = x_embed[:, self.reserve_initials:, :]  # [B, N-reserve, hidden_dim]
            global_feat = torch.mean(patch_tokens, dim=1, keepdim=True)  # [B, 1, hidden_dim]
        else:
            global_feat = torch.mean(x_embed, dim=1, keepdim=True)

        global_feat = global_feat.expand(B, N, -1)  # [B, N, hidden_dim]
        fused_feat = torch.cat([x_embed, global_feat], dim=-1)
        logits = self.out_conv(fused_feat)  # [B, N, block_size * 2]
        logits = logits.view(B, N, self.block_size, 2)
        soft_routing = F.softmax(logits, dim=-1)  # [B, N, block_size, 2]
        
        # 计算router熵（排除reserved tokens）
        routing_probs = soft_routing[:, self.reserve_initials:, :, :]  # [B, N-reserve, block_size, 2]
        router_entropy = -torch.sum(
            routing_probs * torch.log(routing_probs + 1e-8)
        ) / (B * (N - self.reserve_initials) * self.block_size)
        
        # Gumbel-Softmax 生成硬决策 (Training) 或 Argmax (Inference)
        if self.training:
            hard_routing = F.gumbel_softmax(logits, tau=1, hard=True, dim=-1)
        else:
            idx = soft_routing.argmax(dim=-1, keepdim=True)
            hard_routing = torch.zeros_like(soft_routing).scatter_(-1, idx, 1.0)
        
        if self.reserve_initials > 0:
            hard_routing[:, :self.reserve_initials, :, :] = 0
            hard_routing[:, :self.reserve_initials, :, 1] = 1
        
        indices = self._router2indices(hard_routing[:, :, :, 1])

        return hard_routing, indices, router_entropy, soft_routing
    
class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.use_lora = args.use_lora

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=True)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=True)
        
        if self.use_lora:
            self.lora_q = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.n_local_heads)
            self.lora_k = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.n_local_kv_heads)
            self.lora_v = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.n_local_kv_heads)
        
    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with optional asymmetric attention.
        
        Args:
            x: Query tokens [bsz, seqlen_q, dim]
            x_kv: Key/Value tokens [bsz, seqlen_kv, dim]. If None, uses x for KV (standard self-attention)
        """
        if len(x.shape) == 2:
            no_batch = True
            x = x.unsqueeze(0)
            if x_kv is not None:
                x_kv = x_kv.unsqueeze(0)
        else:
            no_batch = False

        bsz, seqlen_q, _ = x.shape
        
        # 如果没有提供x_kv，使用x作为KV（标准self-attention）
        if x_kv is None:
            x_kv = x
        seqlen_kv = x_kv.shape[1]

        if self.use_lora:
            xq = self.wq(x) + self.lora_q(x)
            xk = self.wk(x_kv) + self.lora_k(x_kv)
            xv = self.wv(x_kv) + self.lora_v(x_kv)
        else:
            xq = self.wq(x)
            xk = self.wk(x_kv)
            xv = self.wv(x_kv)

        xq = xq.view(bsz, seqlen_q, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen_kv, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_kv, self.n_local_kv_heads, self.head_dim)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen_kv, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen_kv, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen_q, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, seqlen_kv, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, seqlen_kv, head_dim)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # (bs, n_local_heads, seqlen_q, seqlen_kv)

        scores = F.softmax(scores.float(), dim=-1)

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen_q, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen_q, -1)
        output = self.wo(output)

        if no_batch:
            output = output.squeeze(0)

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
    ):

        super().__init__()
        self.fc1 = Linear(dim, mlp_dim, bias=True)
        self.fc2 = Linear(mlp_dim, dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.act(output)
        output = self.fc2(output)
        return output

class LowRankApproximator(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, dim, bias=False)

        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj(x)
        output = self.up_proj(down)
            
        return output


class BlockPathApproximators(nn.Module):
    def __init__(self, dim: int, rank: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.approximators = nn.ModuleDict()
        total = 2 ** block_size  # block_size位二进制数的总可能数
        full_one_id = total - 1  # 全1的二进制数对应的id（需排除）

        for key in range(total):
            if key == full_one_id:
                continue 
            self.approximators[str(key)] = LowRankApproximator(dim, rank)
        
    def forward(self, 
                x: torch.Tensor, 
                router_indices: torch.Tensor, 
                LRA_mask: torch.Tensor) -> torch.Tensor:
        
        router_indices_squeezed = router_indices.squeeze(-1)
        
        for key in LRA_mask:
            key = key.item()
            key_str = str(key)
            
            if key_str not in self.approximators:
                continue
        
            sub_mask = (router_indices_squeezed == key)  
            if sub_mask.any():
                approximator = self.approximators[key_str]
                x = x.clone()
                x[sub_mask] = approximator(x[sub_mask]) + x[sub_mask]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        # init
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.current_epoch = 0
        self.use_lora = args.use_lora
        self.use_reslr = args.use_reslr

        # attention
        self.attention = Attention(args)
        self.attention_norm = LayerNorm(args.dim, eps=args.norm_eps, use_lora=args.use_lora)

        # ffn
        self.feed_forward = FeedForward(
            dim=args.dim,
            mlp_dim=args.mlp_dim,
        )
        self.ffn_norm = LayerNorm(args.dim, eps=args.norm_eps, use_lora=args.use_lora)
        
        # reslr
        self.dynamic_start_layer = args.dynamic_start_layer
        if self.use_reslr and self.layer_id >= args.dynamic_start_layer:
            self.block_size = args.block_size
            self.is_block_head = (self.layer_id - self.dynamic_start_layer) % self.block_size == 0
            self.current_block_id = (self.layer_id - self.dynamic_start_layer) // self.block_size
            self.block_start_layer = self.dynamic_start_layer + ((self.layer_id - self.dynamic_start_layer) // self.block_size) * self.block_size
            self.current_block_pos = self.layer_id - self.block_start_layer
                
            if self.is_block_head:
                self.router = RouterModule(
                    args.dim, 
                    args.dynamic_router_hdim, 
                    args.dynamic_reserve_initials, 
                    args.norm_eps,
                    block_size=self.block_size,
                    use_lora=args.use_lora
                )
                self.block_path_approximators = BlockPathApproximators(args.dim, args.low_rank_dim, self.block_size)

    def forward(
        self,
        x: torch.Tensor,
        teacher_x: Optional[torch.Tensor] = None,
        block_info: Optional[Dict] = None,
        LRA_mask: Optional[List[List[str]]] = None
    ):
        """
        Args:
            x: student 路径的输入 [batch, seq_len, dim]
            teacher_x: teacher 路径的输入 [batch, seq_len, dim]
            block_info: 块信息字典
            LRA_mask: 低秩近似掩码
        Returns:
            训练时: (teacher_out, student_out, w, block_info)
            推理时: (student_out, w, block_info)
        """
        bsz, seqlen, _ = x.shape
        if block_info is None:
            block_info = {}

        # naive vit or lora or reslr when layerid < dynamic_start_layer
        if not self.use_reslr or self.layer_id < self.dynamic_start_layer:
            w = torch.ones((bsz, seqlen, 1), device=x.device)
            h = x + self.attention(self.attention_norm(x))
            out = h + self.feed_forward(self.ffn_norm(h))
            if self.training:
                # 训练时 teacher 和 student 相同
                return out, out, w, block_info
            else:
                return out, w, block_info

        # block_head need route and approximators
        if self.is_block_head:    
            routing, router_indices, router_entropy, soft_routing = self.router(x)
            block_routing = routing[:, :, :, 1]  # [batch, seq_len, block_size]
            # 提取软概率用于 Ratio Loss (DynamicViT 风格)
            soft_routing_probs = soft_routing[:, :, :, 1]  # [batch, seq_len, block_size] - 保留路径的概率

            # block info
            block_info = {}
            block_info[f"block_{self.current_block_id}_approximators"] = self.block_path_approximators
            block_info[f"block_{self.current_block_id}_routing"] = block_routing
            block_info[f"block_{self.current_block_id}_router_indices"] = router_indices
            block_info[f"block_{self.current_block_id}_router_entropy"] = router_entropy
            block_info[f"block_{self.current_block_id}_soft_routing"] = soft_routing_probs  # 用于 Ratio Loss

        # each layer need approximator or Transformer
        block_path_approximators = block_info[f"block_{self.current_block_id}_approximators"]
        block_routing = block_info[f"block_{self.current_block_id}_routing"]
        router_indices = block_info[f"block_{self.current_block_id}_router_indices"]
        w = block_routing[:, :, self.current_block_pos:self.current_block_pos+1]  # [batch, seq_len, 1]

        # 获取路由掩码
        assert LRA_mask is not None, "LRA_mask must be provided"
        LRA_mask_lora = torch.tensor(LRA_mask[self.current_block_pos][0], device=x.device)
        lora_mask_indices = torch.isin(router_indices.long(), LRA_mask_lora.long())  # [bsz, seqlen, 1]
        LRA_mask_transformer = torch.tensor(LRA_mask[self.current_block_pos][1], device=x.device)
        attention_mask_indices = torch.isin(router_indices.long(), LRA_mask_transformer.long())  # [bsz, seqlen, 1]

        if self.training:
            # ========== Teacher 路径：完整 Transformer 路径 ==========
            if teacher_x is None:
                teacher_x = x
            h_teacher = teacher_x + self.attention(self.attention_norm(teacher_x))
            teacher_out = h_teacher + self.feed_forward(self.ffn_norm(h_teacher))

            # ========== Student 路径：根据 Router 决策混合 ==========
            # 先计算完整 Transformer 输出
            h_student = x + self.attention(self.attention_norm(x))
            transformer_out = h_student + self.feed_forward(self.ffn_norm(h_student))

            # 根据 Router 决策混合：active tokens 用 transformer，inactive tokens 保持原始 x
            student_out = attention_mask_indices * transformer_out + (~attention_mask_indices) * x
            
            # 低秩 approximator 处理 inactive tokens
            student_out = block_path_approximators(student_out, router_indices, LRA_mask_lora)

            return teacher_out, student_out, w, block_info
        else:
            # ========== 推理阶段：非对称 attention ==========
            x_normed = self.attention_norm(x)
            
            # 获取 active tokens 的索引
            active_mask = attention_mask_indices.squeeze(-1)  # [bsz, seqlen]
            
            bsz_local = x.shape[0]
            active_outputs = []
            
            for b in range(bsz_local):
                batch_active_mask = active_mask[b]  # [seqlen]
                x_q = x_normed[b:b+1, batch_active_mask, :]  # [1, num_active, dim]
                x_kv = x_normed[b:b+1]  # [1, seqlen, dim] - 所有 tokens
                
                # 非对称 attention: Q=active, KV=all
                attn_out = self.attention(x_q, x_kv)  # [1, num_active, dim]
                
                # 构建完整输出，active 位置用 attention 输出，inactive 位置用原始 x
                full_attn = x[b:b+1].clone()  # [1, seqlen, dim]
                full_attn[:, batch_active_mask, :] = x[b:b+1, batch_active_mask, :] + attn_out
                active_outputs.append(full_attn)
            
            h = torch.cat(active_outputs, dim=0)  # [bsz, seqlen, dim]
            
            # FFN 只对 active tokens 应用
            h_normed = self.ffn_norm(h)
            ffn_out = self.feed_forward(h_normed)
            output = h + ffn_out
            
            # active tokens 用 transformer 输出，inactive tokens 保持原始 x
            student_out = attention_mask_indices * output + (~attention_mask_indices) * x
            
            # 低秩 approximator 处理 inactive tokens
            student_out = block_path_approximators(student_out, router_indices, LRA_mask_lora)

            return student_out, w, block_info


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.device = params.device

        # embedding layer
        h, w = params.image_size
        fh, fw = params.patch_size
        gh, gw = h // fh, w // fw
        params.num_patches = gh * gw
        self.embedding = nn.Conv2d(3, params.dim, kernel_size=(fh, fw), stride=(fh, fw))
            
        self.cls_token = nn.Parameter(torch.zeros(1, 1, params.dim))
        
        self.pos_embedding = PositionEmbs(params.num_patches, params.dim)

        # criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.criterion_active = ActiveLoss(
            target=params.dynamic_active_target,
            reserve_initials=params.dynamic_reserve_initials
        )
        self.criterion_distill = DistillLoss()

        # transformer-block
        self.n_layers = params.n_layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = LayerNorm(params.dim, eps=params.norm_eps, use_lora=params.use_lora)

        # classifier
        self.classifier = Linear(params.dim, params.num_classes)

        # use_lora
        self.use_lora = params.use_lora
        self.use_reslr = params.use_reslr

        # use_lora to freeze the parameters
        if self.use_lora:
            for name, param in self.named_parameters():
                if (
                    name.startswith('embedding.') or
                    name.startswith('pos_embedding.') or
                    '.feed_forward.' in name or
                    '.attention.wo.' in name or
                    '.attention.wq.' in name or
                    '.attention.wk.' in name or
                    '.attention.wv.' in name
                ):
                    param.requires_grad = False

        # LRA mask
        if self.use_reslr:
            self.LRA_mask = get_indices_from_LRA_mask(params.block_size)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ):
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        if labels.device != device:
            labels = labels.to(device)

        # images embedding
        x = rearrange(self.embedding(x), 'b c h w -> b (h w) c')
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        
        # pos_embedding
        x = self.pos_embedding(x)

        self.acts = []
        self.soft_routing_probs = []  # 收集所有层的软概率用于 Ratio Loss
        self.routing_maps = {}  # 存储每个block-head的路由值
        d_loss = torch.tensor(0.0, device=x.device)
        r_entropy = torch.tensor(0.0, device=x.device)  # 累积 router entropy
        block_info = {}
        
        # 训练时维护两条路径
        teacher_x = x.clone()  # Teacher 路径输入
        student_x = x  # Student 路径输入
        
        for layer in self.layers:
            # reslr
            if self.use_reslr and layer.layer_id >= layer.dynamic_start_layer:
                if self.training:
                    # 训练时：同时维护 teacher 和 student 路径
                    teacher_out, student_out, w, block_info = layer(
                        student_x, teacher_x, block_info, self.LRA_mask
                    )
                    
                    # 计算 D_Loss：每层 cls token 的 MSE
                    student_cls = student_out[:, 0, :]  # [batch, dim]
                    teacher_cls = teacher_out[:, 0, :]  # [batch, dim]
                    layer_d_loss = self.criterion_distill(student_cls, teacher_cls)
                    d_loss += layer_d_loss
                    
                    # 累积 router entropy（只在 block head 层有值）
                    if layer.is_block_head:
                        r_entropy += block_info[f"block_{layer.current_block_id}_router_entropy"]
                        # 收集路由值用于可视化
                        self.routing_maps[layer.current_block_id] = block_info[f"block_{layer.current_block_id}_routing"].detach()
                        # 收集软概率用于 Ratio Loss
                        soft_prob = block_info[f"block_{layer.current_block_id}_soft_routing"]  # [B, N, block_size]
                        self.soft_routing_probs.append(soft_prob)
                    
                    # 更新两条路径的输入
                    teacher_x = teacher_out
                    student_x = student_out
                else:
                    # 推理时：只有 student 路径
                    student_out, w, block_info = layer(student_x, None, block_info, self.LRA_mask)
                    
                    if layer.is_block_head:
                        r_entropy += block_info[f"block_{layer.current_block_id}_router_entropy"]
                        self.routing_maps[layer.current_block_id] = block_info[f"block_{layer.current_block_id}_routing"].detach()
                    
                    student_x = student_out

            # naive vit or lora
            else:
                if self.training:
                    teacher_out, student_out, w, block_info = layer(student_x, teacher_x, block_info)
                    teacher_x = teacher_out
                    student_x = student_out
                else:
                    output = layer(student_x, None, block_info)
                    student_x, w, block_info = output
                
            self.acts.append(w)
        
        # norm
        if self.training:
            teacher_x = self.norm(teacher_x)
            student_x = self.norm(student_x)
        else:
            student_x = self.norm(student_x)

        activation = torch.cat(self.acts, dim=-1)

        # C_Loss 由 student_out 产生，让梯度直接回传到 router
        output = self.classifier(student_x[:, 0])
        self.logits = output
        c_loss = self.criterion(output, labels)
        
        # reslr
        if self.use_reslr:
            # DynamicViT 风格：使用软概率计算 Ratio Loss
            if len(self.soft_routing_probs) > 0:
                # 拼接所有 block 的软概率 [B, N, total_blocks * block_size]
                all_soft_probs = torch.cat(self.soft_routing_probs, dim=-1)
                a_loss = self.criterion_active(all_soft_probs)
            else:
                a_loss = torch.tensor(0.0, device=x.device)
            
            # Metric 继续使用 hard activation (self.acts)
            active_metric = self.criterion_active.metric(activation)

        # naive vit or lora
        else:
            a_loss = None
            active_metric = None
            r_entropy = torch.tensor(0.0, device=x.device)

        return c_loss, a_loss, d_loss, r_entropy, active_metric

if __name__ == '__main__':
    params = ModelArgs(device='cuda')
    model = Transformer(params).to(params.device)
    x = torch.randn(2, 3, 224, 224).to(params.device)
    labels = torch.randint(0, params.num_classes, (2,)).to(params.device) 
    c_loss, a_loss, d_loss, r_entropy, active_metric = model(x, labels)
    print("pass!")