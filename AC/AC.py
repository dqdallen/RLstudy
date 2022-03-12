from model import Actor, Critic, AC_Actor_Loss, AC_Critic_Loss
import gym
import numpy as np
import torch


def train(episode_num, max_len, gamma):
	env = gym.make('CartPole-v1')
	state_num = env.observation_space.shape[0]
	action_num = env.action_space.n
	actor = Actor(state_num, action_num)
	critic = Critic(state_num, action_num)
	aloss = AC_Actor_Loss()
	closs = AC_Critic_Loss()
	aoptim = torch.optim.Adam(actor.parameters(), lr=0.01)
	coptim = torch.optim.Adam(critic.parameters(), lr=0.01)
	for epi in range(episode_num):
		state = env.reset()

		for _ in range(max_len):
			t_state = torch.tensor(state).unsqueeze(0)
			actions = actor(t_state)
			actions = torch.nn.Softmax(dim=1)(actions)
			ac_numpy = actions.detach().numpy()[0]
			action = np.random.choice(np.arange(0, action_num, 1), p=ac_numpy)
			new_state, reward, done, _ = env.step(action)
			if done:
				reward = -1
			else:
				reward = 0.1
			c_v = critic(t_state)
			c_new_v = critic(torch.tensor(new_state).unsqueeze(0))
			delta = reward + gamma * c_new_v - c_v
			c_loss = closs(delta)
			coptim.zero_grad()
			c_loss.backward()
			coptim.step()
			
			deltad = delta.detach()
			a_loss = aloss(actions[(0, torch.tensor(action))], deltad)
			aoptim.zero_grad()
			a_loss.backward()
			aoptim.step()

			state = new_state
			if done:
				break
		
		if (epi + 1) % 100 == 0:
			print(f'epi: {epi}, c_loss: {c_loss}, a_loss: {a_loss}')
			torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()}, './ac_checkpoint.pth')

episode_num=3000
max_len=3000
gamma=0.9
train(episode_num, max_len, gamma)