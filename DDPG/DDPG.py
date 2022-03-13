from model import Actor, Critic, AC_Actor_Loss, AC_Critic_Loss


import gym
import numpy as np
import torch
import random


def train(iter_num, step, replay_size, batch_size, gamma, tau):
	env = gym.make('Pendulum-v1')
	state_num = env.observation_space.shape[0]
	action_num = env.action_space.shape[0]
	a_bound = env.action_space.high
	# print(a_bound);exit()
	actor = Actor(state_num, action_num, a_bound)
	actor_target = Actor(state_num, action_num, a_bound)
	critic = Critic(state_num, action_num)
	critic_target = Critic(state_num, action_num)
	for name, param in actor_target.named_parameters():
		param.requires_grad = False
	for name, param in critic_target.named_parameters():
		param.requires_grad = False
	for name, param in actor_target.named_parameters():
		eval('actor_target.' + name).set_(actor.state_dict()[name])
	for name, param in critic_target.named_parameters():
		eval('critic_target.' + name).set_(critic.state_dict()[name])
	
	

	AcLoss = AC_Actor_Loss()
	CrLoss = AC_Critic_Loss()
	aoptim = torch.optim.Adam(actor.parameters(), lr=0.001)
	coptim = torch.optim.Adam(critic.parameters(), lr=0.002)
	replay = []
	pit = 0
	var = 3
	pre_aloss = float('inf')
	pre_closs = float('inf')
	for epi in range(iter_num):
		state = env.reset()
		for i in range(step):
			state = torch.tensor(state).unsqueeze(0).float()
			actions = actor(state)
			action = actions.detach().numpy()[0]
			action = np.clip(np.random.normal(action, var), -2, 2)

			new_state, reward, done, _ = env.step(action)
			data = (state, action, reward, torch.tensor(new_state).unsqueeze(0), done)
			replay.append(data)
			state = new_state
			
			if len(replay) > replay_size:
				replay.pop(0)
			if len(replay) >= batch_size:
				pit += 1
				var *= .9995
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
				train_newstate = torch.cat(newstate_batch).float()
				reward = torch.tensor(reward_batch).unsqueeze(1).float()
				is_done = torch.tensor(done_batch).unsqueeze(1).float()
				action_tensor = torch.tensor(action_onehot_batch).unsqueeze(1).float()

				pi_target = actor_target(train_newstate)
				q_target = critic_target(train_newstate, pi_target.detach()).detach()
				q_target = reward + gamma * q_target
				q_target[is_done == True] = reward[is_done == True]

				q = critic(train_state, action_tensor)
				closs = CrLoss(q_target, q)
				

				pi = actor(train_state)
				qp = critic(train_state, pi)
				aloss = AcLoss(qp)

				coptim.zero_grad()
				closs.backward()
				coptim.step()
				aoptim.zero_grad()
				aloss.backward()
				aoptim.step()

				if pre_aloss > aloss and pre_closs > closs:
						pre_aloss = aloss
						pre_closs = closs
						torch.save({'aloss': aloss, 'closs': closs, 'actor': actor.state_dict(), 'critic': critic.state_dict()}, './ddpg.pth')
				if pit % 10 == 0:
					ac_sd = actor.state_dict()
					cr_sd = critic.state_dict()
					with torch.no_grad():
						for name, param in actor_target.named_parameters():
						# print(param, name);exit()
							eval('actor_target.' + name).set_( eval('actor.'+name) * tau + param * (1 - tau))
						for name, param in critic_target.named_parameters():
							eval('critic_target.' + name).set_( eval('critic.'+name) * tau + param * (1 - tau))

				if pit % 10000 == 0:
					print(f'epi: {epi}, closs: {closs}, aloss: {aloss}')
			if done:
				break
	torch.save({'aloss': aloss, 'closs': closs, 'actor': actor.state_dict(), 'critic': critic.state_dict()}, './ddpgfinal.pth')
				

iter_num = 2000
step=200 
replay_size = 10000
batch_size = 32
gamma = 0.9
tau = 0.01
train(iter_num, step, replay_size, batch_size, gamma, tau)
