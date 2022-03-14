import gym
import torch
from model import Actor, Critic
import time


env = gym.make('CartPole-v1')
actor = Actor(4, 2)
actor.load_state_dict(torch.load('ppofinal.pth')['actor'])

state = env.reset()
env.render()
#time.sleep(10)

for i in range(300):
	state = torch.tensor(state).unsqueeze(0).float()

	action_q = actor(state)[0].detach().numpy()
	ac = action_q.argmax()
	state, reward, done, _ = env.step(ac)
	env.render()
	time.sleep(0.01)
	if done:
		print(i, state, reward, done)
		break
env.close()