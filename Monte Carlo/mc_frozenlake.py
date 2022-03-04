import gym
import numpy as np

env = gym.make("FrozenLake-v0", is_slippery=False) # FrozenLake-v1 FrozenLake-v0都可以，看库的版本

def on_policy_mc_mg(episode_num, env, gamma, first_every, eps):
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    cnt = np.zeros_like(policy)  # 记录不同的g出现的次数
    q = np.zeros_like(policy)  # 存储过程中的q值，用于计算最终q值
    for epi in range(episode_num):  # 迭代episode_num次，生成episode_num个序列
        g = 0  # 即G，累积回报
        state = env.reset()  # 重置环境
        a_s = []  # 存储一个序列中的动作和状态
        while True:
            action = np.random.choice(env.action_space.n, p=policy[state])  # 根据策略policy随机选取动作（这里采用随机是因为要探索）
            new_state, reward, done, info = env.step(action)  # 放到环境中执行后，得到新状态和奖励，done表示是否结束
            # 这里对原环境产生的奖励做了修改，到达终点则奖励200，到达陷阱则-200，走出边界则-200并结束，向前走一步为-1
            # 原始环境中走到边界是可以继续走的，这里进行了简单修改，方便训练，并且走一步奖励-1使得可以走的步数尽量少。
            if reward == 1:
                reward = 200
            elif reward == 0 and done:
                reward = -200
            elif reward == 0 and state == new_state:
                reward = -200
                done = True
            else:
                reward = -1 
            if first_every == 'every':  # 是every visit 还是first vist
                a_s.append((action, state, reward))

            else:
                if (action, state, reward) not in a_s:
                    a_s.append((action, state, reward))
            state = new_state
            if done:
                break
        # 上述while True得到一条序列后，序列中状态s对应的q值
        g = 0
        for a, s, r in reversed(a_s): # reversed是倒序，使得在计算g的时候距离当前状态越远的系数越小，起到加权的作用
            g = r + gamma * g
            cnt[s][a] += 1
            q[s][a] += (g - q[s][a]) / cnt[s][a]
            ac = np.argmax(q[s]) # 找到q值最大的对应的动作
            # 保留探索部分的概率
            policy[s] = eps / env.action_space.n
            policy[s][ac] = 1 - eps + eps / env.action_space.n

        if (epi + 1) % 10000 == 0:
            print(f'epi: {epi}, g = {g}')
    return policy, q       

def off_policy_mc_mg(env, gamma, episode_num):
    # 初始化两个策略，分别是目标策略和行为策略
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    cnt = np.zeros_like(policy)
    q = np.zeros_like(policy)
    
    for epi in range(episode_num):
        g = 0
        state = env.reset()
        a_s = []
        w = 1
        while True:
            action = np.random.choice(env.action_space.n, p=behavior_policy[state])
            new_state, reward, done, _ = env.step(action)
            if reward == 1:
                reward = 200
            elif reward == 0 and done:
                reward = -200
            elif reward == 0 and state == new_state:
                reward = -200
                done = True
            else:
                reward = -1 
            
            a_s.append((action, state, reward))
            state = new_state
            if done:
                break
        # w这个变量用于计算重要性采样中的权重
        g = 0
        for a, s, r in reversed(a_s):
            g =  r + gamma * g
            cnt[s][a] += w
            q[s][a] += w / cnt[s][a] * (g - q[s][a])
            ac = np.argmax(q[s])
            # 这里和on_policy不一样，这里产生行为的是包含探索的，即上面的random.choice，这里做决策的时候，采用贪心策略
            policy[s] = 0
            policy[s][ac] = 1
            if ac != a:
                break
        w = w / behavior_policy[s][a]
        if (epi + 1) % 10000 == 0:
            print(f'epi = {epi}, g = {g}')
    return policy, q

if __name__ == '__main__':
    gamma = 0.9
    eps = 0.1
    episode_num = 50000
    method = 'off'
    if method == 'on':
        first_every = 'first'
        p, q = on_policy_mc_mg(episode_num, env, gamma, first_every, eps)
    else:
        p, q = off_policy_mc_mg(env, gamma, episode_num)
    # test
    state = env.reset()
    g = 0
    i = 0
    import time
    while True:
        i += 1
        action = p[state].argmax()
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)
        g = reward
        if done:
            break