from model import PG, PGLoss
import gym
import numpy as np
import torch


def train(episode_num, max_len, gamma):
	env = gym.make('MountainCar-v0')
	state_num = env.observation_space.shape[0]
	action_num = env.action_space.n
	pg = PG(state_num, action_num)
	pgloss = PGLoss()
	optim = torch.optim.Adam(pg.parameters(), lr=0.01)
	for epi in range(episode_num):
		state = env.reset()
		mc_buffer = []
		for _ in range(max_len):
			t_state = torch.tensor(state).unsqueeze(0)
			actions = pg(t_state).detach()
			actions = torch.nn.Softmax(dim=1)(actions).numpy()[0]
			action = np.random.choice(np.arange(0, action_num, 1), p=actions)
			
			new_state, reward, done, _ = env.step(action)
			mc_buffer.append((state, action, reward))
			state = new_state
			if done:
				break
		v = []
		vt = 0
		pi = []
		for s, a, r in reversed(mc_buffer):
			vt = vt * gamma + r
			v.append(vt)
			s = torch.tensor(s).unsqueeze(0)
			actions = pg(s)
			soft_ac = torch.nn.Softmax(1)(actions)
			pi.append(soft_ac)
		pi = torch.cat(pi)
		v = torch.tensor(v).unsqueeze(1)
		loss = pgloss(soft_ac, v)
		optim.zero_grad()
		loss.backward()
		optim.step()
		if (epi + 1) % 100 == 0:
			print(f'epi: {epi}, loss: {loss}')
			torch.save(pg.state_dict(), './pg_checkpoint.pth')

episode_num=3000
max_len=3000
gamma=0.9
# train(episode_num, max_len, gamma)
env = gym.make('MountainCarContinuous-v0')  # Continuous
env.reset()
print(env.action_space)
print(env.observation_space)
print(env.step(0))