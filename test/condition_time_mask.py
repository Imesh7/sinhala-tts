

# Test function

import torch
from zipvoice.utils.common import condition_time_mask


val = torch.tensor((10, 10, 10, 6, 8))
res = condition_time_mask(val, (0.7, 0.9))


print(f"val shape {val.shape}")
print(res)          # (2, 150)
print(res[0].any())       # True (unless mask_size=0)
print(res[0].sum())