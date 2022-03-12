import gym
import torch
import torch.nn as nn
import random
import numpy as np
from model import A3C, Actor_Loss, Critic_Loss, Actor, Critic


class Worker(nn.Module):
	def __init__(self, iter_num, gamma, eps, replay_size, batch_size):
		self.iter_num = iter_num, 
		self.gamma = gamma, 
		self.eps = eps, 
		self.replay_size = replay_size, 
		self.batch_size = batch_size
		self.replay = [] # 存储经验回放
		self.env = gym.make('CartPole-v1')
		self.state_num = self.env.observation_space.shape[0]
		self.action_num = self.env.action_space.n
		self.a3c = A3C(self.state_num, self.action_num)
		self.actor = Actor()
		self.critic = Critic()
		self.step = 300 # 一定次数后，就算没到终止也reset，然后重新初始状态
		self.aloss = Actor_Loss(c=0.1)
		self.closs = Critic_Loss()
		self.aoptim = torch.optim.Adam(actor.parameters(), lr=0.0001)
		self.coptim = torch.optim.Adam(critic.parameters(), lr=0.0001)
		self.aloss_sum = 0
		self.closs_sum = 0

	def train(self):
		pit = 0
		# 迭代iter_num次
		for i in range(self.iter_num):
			state = self.env.reset()
			xl_buffer = []
			for j in range(self.step):
				# 根据现有状态，经过DQN得到每个动作对应的q值
				state = torch.tensor(state).unsqueeze(0)
				action_q = self.actor(state).detach().numpy()
				# 通过e-greedy得到动作
				action = np.random.choice(np.arange(0, action_num, 1), p=action_q)

				# 执行动作
				new_state, reward, done, _ = env.step(action)
				# 修改reward，来促进训练，希望维持不倒，所以非done时为正

				# 存储经验回放，把五元组存下来
				action_one_hot = torch.zeros((1, action_num))
				action_one_hot[0, action] = 1
				data = (state, action_one_hot, reward, torch.tensor(new_state).unsqueeze(0), done)
				xl_buffer.append(data)
				
				if done:
					break
				state = new_state
			if done:
				q = 0
			else:
				q = self.critic(torch.tensor(state).unsqueeze(0))
			for xl in range(len(xl_buffer) - 1, -1, -1):
				q = xl_buffer[xl][2] + self.gamma * q
				vt = self.critic(xl_buffer[xl][0])
				a_loss = self.aloss(pi, q.detach(), vt.detach(), xl_buffer[xl][1])
				self.aloss_sum += a_loss
				c_loss = self.closs(q, vt)
				self.closs_sum += c_loss
			

		# 保存参数
		torch.save(dqn.state_dict(), './dqn_catpole.pth')