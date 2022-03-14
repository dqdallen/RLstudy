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
		self.softmax = nn.Softmax(1)

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		x = self.l2(x)
		actions = self.softmax(x)
		return actions

class Critic(nn.Module):
	def __init__(self, state_num, action_num):
		super(Critic, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 20)
		self.l11 = nn.Linear(action_num, 20)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(20, 1)

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		v = self.l2(x)
		return v


# 均方误差
class AC_Actor_Loss(nn.Module):
	def __init__(self):
		super(AC_Actor_Loss, self).__init__()

	def forward(self, pi, pi_k, A, eps):
		left = pi / (pi_k + 1e-5) * A
		right = torch.clamp(pi / (pi_k + 1e-5), 1-eps, 1+eps) * A
		l_r = torch.cat((left.unsqueeze(1), right.unsqueeze(1)), 1)
		loss = torch.min(l_r, 1)[0]

		return -loss.mean()

class AC_Critic_Loss(nn.Module):
	def __init__(self):
		super(AC_Critic_Loss, self).__init__()

	def forward(self, q, v):
		loss = (q - v) ** 2
		return loss.mean()