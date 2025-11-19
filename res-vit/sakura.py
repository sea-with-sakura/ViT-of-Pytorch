import torch
block_size = 2
total = 2 ** block_size  # block_size位二进制数的总可能数
full_one_id = total - 1  # 全1的二进制数对应的id（需排除）

for key in range(total):
    if key == full_one_id:
        continue 
    print(key)
