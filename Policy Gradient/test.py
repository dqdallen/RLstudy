import gym
import torch
from model import PG
import time


path = './pg_checkpoint.pth'  # './dqn_catpole.pth'./NatureDQN_checkpoint/NatureDQN.pth
pg = PG(4, 2)
pg.load_state_dict(torch.load(path))
env = gym.make('CartPole-v1')
state = env.reset()
env.render()
#time.sleep(10)

for i in range(300):
	state = torch.tensor(state).unsqueeze(0)
	action_q = pg(state).detach().numpy()
	ac = action_q[0].argmax()
	state, reward, done, _ = env.step(ac)
	env.render()
	time.sleep(0.01)
	if done:
		print(i, state, reward, done)
		break
env.close()
