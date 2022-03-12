import torch
import torch.nn as nn

# 两层fc构建神经网络
class A3C(nn.Module):
	def __init__(self, state_num, action_num):
		super(A3C, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 20)
		self.relu = nn.ReLU()
		# self.l2 = nn.Linear(20, action_num)
		# self.l3 = nn.Linear(20, 1)

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		# pi = self.l2(x)
		# v = self.l3(x)
		return x


class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(20, 4)

	def forward(self, a3c, state):
		x = a3c(state)
		pi = self.l1(x)
		return pi

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(20, 1)

	def forward(self, a3c, state):
		x = a3c(state)
		v = self.l1(x)
		return v

# 均方误差
class Actor_Loss(nn.Module):
	def __init__(self, c):
		super(Actor_Loss, self).__init__()
		self.c = c

	def forward(self, pi, q, v, a):
		h = pi * torch.log(pi)
		h = torch.sum(h, 1) * -1
		loss = -torch.log(pi[a == 1]) * (v - q) - slef.c * h
		return loss.mean()

class Critic_Loss(nn.Module):
	def __init__(self):
		super(Critic_Loss, self).__init__()

	def forward(self, q, v):
		loss = (q - v) ** 2
		return loss.mean()