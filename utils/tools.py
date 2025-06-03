import numpy as np
import torch
import random

def random_seed(n):
    # n = int(n)
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)

    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True  # 设置CUDNN为deterministic  
    # torch.backends.cudnn.benchmark = False  # 关闭CUDNN的自动调优，确保每次运行都一致
