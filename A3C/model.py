import torch
import torch.nn as nn

# 两层fc构建神经网络
class A3C(nn.Module):
	def __init__(self, state_num, action_num):
		super(A3C, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 200)
		self.relu = nn.ReLU()
		self.l_a = nn.Linear(200, action_num)
		self.softmax = torch.nn.Softmax(1)
		self.l_q = nn.Linear(200, 1)


	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		a = self.l_a(x)
		a = self.softmax(a)
		q = self.l_q(x)
		return a, q


class Actor_Loss(nn.Module):
	def __init__(self, c):
		super(Actor_Loss, self).__init__()
		self.c = c

	def forward(self, pi, q, v, a):
		h = pi * torch.log(pi)
		h = torch.sum(h, 1) * -1
		loss = -torch.log(pi[a == 1]) * (q - v) - self.c * h
		return loss.mean()


class Critic_Loss(nn.Module):
	def __init__(self):
		super(Critic_Loss, self).__init__()

	def forward(self, q, v):
		loss = (q - v) ** 2
		return loss.mean()