import torch
from model_utils import get_indices_from_LRA_mask,_gen_LRA_mask

mask = get_indices_from_LRA_mask(1)
print(mask)