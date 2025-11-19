import torch

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def _gen_LRA_mask(block_size: int):
    LRA_mask = []
    for j in range(block_size):
        layer_for_lora = [(i, j) for i in range(j + 1)]
        part_a = [(i, jp) for jp in range(j) for i in range(jp + 1)]
        part_b = [(i, jp) for jp in range(j + 1, block_size) for i in range(j + 1, jp + 1)]
        layer_for_transformer = part_a + part_b
        layer_for_ste = [(i, jp) for jp in range(j + 1, block_size) for i in range(j + 1)]
        LRA_mask.append((layer_for_lora, layer_for_transformer, layer_for_ste))
    return LRA_mask


mapping_table_2 = [
    [
        [1],      # 00 
        [0],      # 01
    ],        
    [
        [],       # 10
        [2],      # 11
    ]
]

mapping_table_4 = [
    [
        [4,5,6,7],  # 00 
        [2,3],      # 01
        [1],        # 02
        [0]         # 03
    ],        
    [
        [],         # 10
        [10,11],    # 11
        [9],        # 12
        [8]         # 13
    ],
    [
        [],         # 20
        [],         # 21
        [13,5],     # 22
        [12,4],     # 23
    ],
    [
        [],         # 30
        [],         # 31
        [],         # 32 
        [2,6,10,14],# 33
    ],
]


def get_indices_from_LRA_mask(block_size, mapping_table=None):

    if mapping_table is None:
        if block_size == 2:
            mapping_table = mapping_table_2
        elif block_size == 4:
            mapping_table = mapping_table_4
        else:
            raise ValueError(f"不支持的block_size: {block_size}，当前仅支持2和4")
    
    lra_mask = _gen_LRA_mask(block_size)
    result = []
    for j in range(block_size):
        lora_coords, transformer_coords, ste_coords = lra_mask[j]
        
        # lora
        lora_mapping = []
        for i, jp in lora_coords:
            lora_mapping.extend(mapping_table[i][jp])
        lora_mapping = sorted(list(set(lora_mapping)))  # 去重+排序
        
        # transformer
        transformer_mapping = []
        for i, jp in transformer_coords:
            transformer_mapping.extend(mapping_table[i][jp])
        block_mask = (1 << block_size) - 1  
        transformer_mapping.append(block_mask)  
        transformer_mapping = sorted(list(set(transformer_mapping)))  # 去重+排序
        
        # ste
        ste_mapping = []
        for i, jp in ste_coords:
            ste_mapping.extend(mapping_table[i][jp])
        ste_mapping = sorted(list(set(ste_mapping)))  # 去重+排序
        
        result.append((lora_mapping, transformer_mapping, ste_mapping))
    return result