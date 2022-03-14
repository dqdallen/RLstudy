import multiprocessing as mp
import time
import os
import torch

a = torch.tensor([[1,2], [3,4]])
b = torch.tensor([[3,4]])
print(torch.max(a, 1))
