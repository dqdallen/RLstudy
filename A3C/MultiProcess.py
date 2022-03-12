import multiprocessing as mul
import time

def func1(x):
	print(x)

def func2(x):
	print(x)

print(mul.cpu_count())

import torch as t
from torch.autograd import Variable as v

a = v(t.FloatTensor([2, 3]), requires_grad=True)    
b = a + 3
c = b * b * 3
out = c#.mean()
out.backward(gradient=t.Tensor([2,1]), retain_graph=True) # 这里可以不带参数，默认值为‘1’，由于下面我们还要求导，故加上retain_graph=True选项

print(a.grad) # tensor([15., 18.])
print(a)