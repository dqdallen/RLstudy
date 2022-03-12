import gym
import torch
from model import Actor, Critic
import time
import io


path = './ddpg.pth'
env = gym.make('Pendulum-v1')
state_num = env.observation_space.shape[0]
action_num = env.action_space.shape[0]
actor = Actor(state_num, action_num, env.action_space.high)
critic = Critic(state_num, action_num)

state_dict = torch.load(path)
actor.load_state_dict(state_dict['actor'])
critic.load_state_dict(state_dict['critic'])

state = env.reset()
env.render()
#time.sleep(10)

for i in range(300):
	state = torch.tensor(state).unsqueeze(0).float()
	action_q = actor(state).detach().numpy()
	ac = action_q[0]
	# q = critic(state, action_q)

	state, reward, done, _ = env.step(ac)
	# print(reward)
	env.render()
	time.sleep(0.01)
	if done:
		print(i, state, reward, done)
		break
env.close()
