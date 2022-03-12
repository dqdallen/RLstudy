import gym
import torch
from model import Actor, Critic
import time


path = './ac_checkpoint.pth'  # './dqn_catpole.pth'./NatureDQN_checkpoint/NatureDQN.pth
actor = Actor(4, 2)
critic = Critic(4, 2)
state_dict = torch.load(path)

actor.load_state_dict(state_dict['actor'])
critic.load_state_dict(state_dict['critic'])
env = gym.make('CartPole-v1')
state = env.reset()
env.render()
#time.sleep(10)

for i in range(300):
	state = torch.tensor(state).unsqueeze(0)
	action_q = actor(state).detach().numpy()
	ac = action_q[0].argmax()
	state, reward, done, _ = env.step(ac)
	env.render()
	time.sleep(0.01)
	if done:
		print(i, state, reward, done)
		break
env.close()
