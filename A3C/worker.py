import gym
import torch
import torch.nn as nn
import random
import numpy as np
from model import A3C, Actor_Loss, Critic_Loss
import os


class Worker:
	def __init__(self, iter_num, gamma, step):
		self.iter_num = iter_num
		self.gamma = gamma 
		self.replay = [] # 存储经验回放
		self.env = gym.make('CartPole-v1')
		self.state_num = self.env.observation_space.shape[0]
		self.action_num = self.env.action_space.n
		self.a3c = A3C(self.state_num, self.action_num)
		# self.actor = Actor(self.a3c)
		# self.critic = Critic(self.a3c)
		self.step = step # 一定次数后，就算没到终止也reset，然后重新初始状态
		self.aloss = Actor_Loss(c=0.001)
		self.closs = Critic_Loss()

		# self.aloss_sum = 0
		# self.closs_sum = 0

	def train(self, a3c, a3c_op, ite, pre_loss):
		pit = 0
		# 迭代iter_num次

		while ite[0] < self.iter_num:
			state = self.env.reset()
			xl_buffer = []
			for j in range(self.step):
				ite[0] += 1
				state = torch.tensor(state).unsqueeze(0)
				action_q = self.a3c(state)[0].detach().numpy()[0]

				action = np.random.choice(np.arange(0, self.action_num, 1), p=action_q)

				# 执行动作
				new_state, reward, done, _ = self.env.step(action)
				# 修改reward，来促进训练，希望维持不倒，所以非done时为正

				# 存储经验回放，把五元组存下来
				action_one_hot = torch.zeros((1, self.action_num))
				action_one_hot[0, action] = 1
				data = (state, action_one_hot, reward, torch.tensor(new_state).unsqueeze(0), done)
				xl_buffer.append(data)
				
				if done:
					break
				state = new_state
			if done:
				q = torch.tensor(0.)
			else:
				q = self.a3c(torch.tensor(state).unsqueeze(0))[1][0]
			aloss_sum = 0
			closs_sum = 0
			loss_sum = 0
			for xl in range(len(xl_buffer) - 1, -1, -1):
				q = xl_buffer[xl][2] + self.gamma * q
				vt = self.a3c(xl_buffer[xl][0])[1]
				pi = self.a3c(xl_buffer[xl][0])[0]
				a_loss = self.aloss(pi, q.detach(), vt.detach(), xl_buffer[xl][1])
				c_loss = self.closs(q.detach(), vt)
				loss_sum = loss_sum + a_loss + c_loss
			loss_sum = loss_sum / len(xl_buffer)
			a3c_op.zero_grad()
			loss_sum.backward()
			for p, shared_p in zip(self.a3c.parameters(), a3c.parameters()):
				shared_p.grad = p.grad

			a3c_op.step()

			with torch.no_grad():
				for name, param in self.a3c.named_parameters():
					eval('self.a3c.' + name).set_(a3c.state_dict()[name])
			if loss_sum < pre_loss[0]:
				pre_loss[0] = loss_sum.data
				torch.save(a3c.state_dict(), './a3c_ckpt.pth')
			if ite % 100 == 0:
				print(f'pid: {os.getpid()}, epi: {ite}, loss_sum: {loss_sum}')