import gym
import torch
from model import A3C
import time


env = gym.make('CartPole-v1')
a3c = A3C(env.observation_space.shape[0], env.action_space.n)
a3c.load_state_dict(torch.load('a3c_ckpt.pth'))
state = env.reset()
env.render()
#time.sleep(10)

for i in range(300):
	state = torch.tensor(state).unsqueeze(0)
	action_q = a3c(state)[0].detach().numpy()
	ac = action_q[0].argmax()
	state, reward, done, _ = env.step(ac)
	env.render()
	time.sleep(0.01)
	if done:
		print(i, state, reward, done)
		break
env.close()