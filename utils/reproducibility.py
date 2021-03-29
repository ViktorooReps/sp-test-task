import torch
import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)