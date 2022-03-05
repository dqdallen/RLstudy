import torch
import torch.nn as nn

# 两层fc构建神经网络
class DQN(nn.Module):
	def __init__(self, state_num, action_num):
		super(DQN, self).__init__()
		self.state_num = state_num
		self.l1 = nn.Linear(state_num, 20)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(20, action_num)

	def forward(self, state):
		x = self.l1(state)
		x = self.relu(x)
		action_q = self.l2(x)
		return action_q


# 均方误差
class DQNLoss(nn.Module):
	def __init__(self):
		super(DQNLoss, self).__init__()

	def forward(self, now_q, pre_q):
		error = (now_q - pre_q) ** 2
		return torch.mean(error)