from model import Actor, Critic, AC_Actor_Loss, AC_Critic_Loss

import gym
import numpy as np
import torch
import random


def train(iter_num, step, total_num, gamma, eps, batch_size):
	env = gym.make('CartPole-v1')
	state_num = env.observation_space.shape[0]
	action_num = env.action_space.n

	actor = Actor(state_num, action_num)
	critic = Critic(state_num, action_num)


	AcLoss = AC_Actor_Loss()
	CrLoss = AC_Critic_Loss()
	aoptim = torch.optim.Adam(actor.parameters(), lr=0.001)
	coptim = torch.optim.Adam(critic.parameters(), lr=0.002)
	pre_aloss = float('inf')
	pre_closs = float('inf')
	for it in range(iter_num):
		pit = 0

		state_batch = []
		action_onehot_batch = []
		reward_batch = []
		newstate_batch = []
		done_batch = []
		actions_batch = []
		values_batch = []
		# with torch.no_grad():
		# 	for name, param in actor_target.named_parameters():
		# 		eval('actor_target.' + name).set_(eval('actor.'+name))
		while pit < total_num:
			state = env.reset()
			for i in range(step):
				pit += 1
				state = torch.tensor(state).unsqueeze(0).float()
				actions = actor(state).detach()

				action = np.random.choice(np.arange(0, action_num, 1), p=actions.numpy()[0])

				new_state, reward, done, _ = env.step(action)

				action_one_hot = torch.zeros((1, action_num))
				action_one_hot[0, action] = 1
				v = critic(state)
				values_batch.append(v)
				state_batch.append(state)
				action_onehot_batch.append(action_one_hot)
				reward_batch.append(reward)
				newstate_batch.append(torch.tensor(new_state).unsqueeze(0))
				done_batch.append(done)
				actions_batch.append(actions)
				state = new_state
				if done:
					break
		values_tensor = torch.tensor(values_batch).unsqueeze(1).float()
		train_state = torch.cat(state_batch).float()
		train_newstate = torch.cat(newstate_batch).float()
		reward = torch.tensor(reward_batch).unsqueeze(1).float()
		is_done = torch.tensor(done_batch).unsqueeze(1).float()
		action_tensor = torch.cat(action_onehot_batch)
		pi_k = torch.cat(actions_batch).float()
		
		q = 0
		q_batch = []
		adv_batch = []
		for i in range(reward.shape[0]-1, -1, -1):
			q = gamma * q * (1-is_done[i][0]) + reward[i][0]
			q_batch.append(q)
			adv_batch.append(q - values_tensor[i][0])

		q_tensor = torch.tensor(q_batch).unsqueeze(1)
		adv_tensor = torch.tensor(adv_batch).unsqueeze(1)
		q_tensor = torch.flip(q_tensor, dims=[0])
		adv_tensor = torch.flip(adv_tensor, dims=[0])
		for kk in range(5):
			ind = random.sample(list(range(reward.shape[0])), batch_size)
			pi = actor(train_state[ind])
			v = critic(train_state[ind])
			closs = CrLoss(q_tensor[ind], v)
			aloss = AcLoss(pi[action_tensor[ind] == 1], pi_k[ind][action_tensor[ind] == 1], adv_tensor[ind].squeeze(1), eps)
			
			coptim.zero_grad()
			closs.backward()
			coptim.step()

			aoptim.zero_grad()
			aloss.backward()
			aoptim.step()

		if pre_aloss > aloss and pre_closs > closs:
			pre_aloss = aloss
			pre_closs = closs
			torch.save({'aloss': aloss, 'closs': closs, 'actor': actor.state_dict(), 'critic': critic.state_dict()}, './ppo.pth')


		if (it + 1) % 100 == 0:
			print(f'iter: {it}, closs: {closs}, aloss: {aloss}')
	torch.save({'aloss': aloss, 'closs': closs, 'actor': actor.state_dict(), 'critic': critic.state_dict()}, './ppofinal.pth')
				

iter_num = 2000
total_num = 1000
step = 200 
gamma = 0.9
eps = 0.2
batch_size = 32
train(iter_num, step, total_num, gamma, eps, batch_size)