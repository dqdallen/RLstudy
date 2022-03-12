import torch
import torch.nn as nn

# 两层fc构建神经网络
class Actor(nn.Module):
	def __init__(self, state_num, action_num, a_bound):
		super(Actor, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 30)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(30, action_num)
		self.tanh = nn.Tanh()
		self.a_bound = torch.tensor(a_bound[0])

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		x = self.l2(x)
		action = self.tanh(x)
		return action * self.a_bound

class Critic(nn.Module):
	def __init__(self, state_num, action_num):
		super(Critic, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 30)
		self.l2 = nn.Linear(action_num, 30)
		self.relu = nn.ReLU()
		self.l3 = nn.Linear(30, 1)

	def forward(self, state, action):
		sx = self.l1(state)
		ax = self.l2(action)
		x = self.relu(sx + ax)
		q = self.l3(x)
		return q


# 均方误差
class AC_Actor_Loss(nn.Module):
	def __init__(self):
		super(AC_Actor_Loss, self).__init__()

	def forward(self, qp):
		loss = -qp
		return loss.mean()

class AC_Critic_Loss(nn.Module):
	def __init__(self):
		super(AC_Critic_Loss, self).__init__()

	def forward(self, q_target, q):
		loss = (q_target - q) ** 2
		return loss.mean()