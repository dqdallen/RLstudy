import torch
import torch.nn as nn

# 两层fc构建神经网络
class Actor(nn.Module):
	def __init__(self, state_num, action_num):
		super(Actor, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 20)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(20, action_num)

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		action_q = self.l2(x)
		return action_q

class Critic(nn.Module):
	def __init__(self, state_num, action_num):
		super(Critic, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 20)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(20, 1)

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		action_q = self.l2(x)
		return action_q


# 均方误差
class AC_Actor_Loss(nn.Module):
	def __init__(self):
		super(AC_Actor_Loss, self).__init__()

	def forward(self, pi, v):
		loss = -torch.log(pi) * v
		return loss.mean()

class AC_Critic_Loss(nn.Module):
	def __init__(self):
		super(AC_Critic_Loss, self).__init__()

	def forward(self, delta):
		loss = delta ** 2
		return loss.mean()