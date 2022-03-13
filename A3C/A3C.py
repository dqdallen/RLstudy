from worker import Worker
import multiprocessing as mp
from model import A3C, Actor_Loss, Critic_Loss
import gym
import torch
import time


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

if __name__ == '__main__':

	iter_num = 50000
	gamma = 0.9
	env = gym.make('CartPole-v1')
	state_num = env.observation_space.shape[0]
	action_num = env.action_space.n

	a3c = A3C(state_num, action_num).share_memory()
	a3c_op = SharedAdam(a3c.parameters())


	j = torch.LongTensor([0]).share_memory_()
	pre_loss = torch.FloatTensor([float('inf')]).share_memory_()
	step = 200
	# p = mp.Pool(2)

	pools = []
	ite = mp.Value('i', 0)
	for i in range(4):
		worker = Worker(iter_num, gamma, step)
		p = mp.Process(target=worker.train, args=(a3c, a3c_op, j, pre_loss))
		pools.append(p)
		# p.apply_async(worker.train, args=(share_dic, alr, clr, lock))
	for i in pools:
		i.start()
	for i in pools:
		i.join()
	torch.save(a3c.state_dict(), './a3c_ckptfinal.pth')
	