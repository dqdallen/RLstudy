import gym
import torch
import torch.nn as nn
import random
import numpy as np
from model import DQN, DQNLoss


# 原始DQN目标Q和优化的Q用同一个网络计算，存在较强依赖性
def train(iter_num, gamma, eps, replay_size, batch_size):
	replay = [] # 存储经验回放
	env = gym.make('CartPole-v1')
	state_num = 4
	action_num = env.action_space.n
	dqn = DQN(state_num, action_num)
	dqn_target = DQN(state_num, action_num)
	step = 300 # 一定次数后，就算没到终止也reset，然后重新初始状态
	dqnloss = DQNLoss()
	optim = torch.optim.Adam(dqn.parameters(), lr=0.0001)
	pit = 0
	# 迭代iter_num次
	for i in range(iter_num):
		state = env.reset()
		for j in range(step):
			# 根据现有状态，经过DQN得到每个动作对应的q值
			state = torch.tensor(state).unsqueeze(0)
			action_q = dqn(state).detach().numpy()
			# 通过e-greedy得到动作
			if random.random() <= eps / action_num:
				action = np.random.choice(np.arange(0, action_num, 1))
			else:
				action = np.argmax(action_q, 1)[0]
			eps -= (0.5 - 0.01) / 10000
			eps = max(eps, 0.01)
			# 执行动作
			new_state, reward, done, _ = env.step(action)
			# 修改reward，来促进训练，希望维持不倒，所以非done时为正
			if done:
				reward = -1
			else:
				reward = 0.1
			# 存储经验回放，把五元组存下来
			action_one_hot = torch.zeros((1, action_num))
			action_one_hot[0, action] = 1
			data = (state, action_one_hot, reward, torch.tensor(new_state).unsqueeze(0), done)
			replay.append(data)
			if len(replay) > replay_size:
				replay.pop(0)
			if len(replay) > batch_size:
				# 随机抽取batch size个样本用于训练
				# 将其中的元素组成batch size大小的tensor
				data = random.sample(replay, batch_size)
				state_batch = []
				action_onehot_batch = []
				reward_batch = []
				newstate_batch = []
				done_batch = []
				for k in range(batch_size):
					state_batch.append(data[k][0])
					action_onehot_batch.append(data[k][1])
					reward_batch.append(data[k][2])
					newstate_batch.append(data[k][3])
					done_batch.append(data[k][4])
				train_state = torch.cat(state_batch)
				train_newstate = torch.cat(newstate_batch)
				reward = torch.tensor(reward_batch)
				is_done = torch.tensor(done_batch)
				action_onehot = torch.cat(action_onehot_batch)
				# 计算目标r+gamma * Q
				aciton_q = dqn_target(train_newstate)
				now_q = reward + gamma * torch.max(aciton_q, 1)[0]
				# 终点的情况下不乘gamma
				now_q[is_done == True] = reward[is_done == True]
				# 计算损失
				pre_q = dqn(train_state)
				loss = dqnloss(now_q, pre_q[action_onehot == 1])
				pit += 1
				if pit % 10 == 0:
					torch.save(dqn.state_dict(), './NatureDQN_checkpoint/NatureDQN.pth')
					dqn_target.load_state_dict(torch.load('./NatureDQN_checkpoint/NatureDQN.pth'))
				if pit % 10000 == 0:
					print(f'iter: {i}, train_times: {pit}, loss: {loss}, qs: {now_q.sum()}')
				# 更新参数
				optim.zero_grad()
				loss.backward()
				optim.step()
			if done:
				break
			state = new_state
	# 保存参数
	# torch.save(dqn.state_dict(), './dqn_catpole.pth')

iter_num=1000
gamma = 0.9
eps=0.5
replay_size=10000
batch_size=32
train(iter_num, gamma, eps, replay_size, batch_size)
