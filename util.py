import pandas as pd
import numpy as np
import os
import math
import random
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

def set_random_seed(seed=418):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

