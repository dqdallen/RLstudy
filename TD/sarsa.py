import gym
import numpy as np
import time


env = gym.make("Taxi-v2")

def sarsa(episode_num, env, gamma, eps):
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    q = np.zeros_like(policy)
    alpha = 0.1
    for epi in range(episode_num):
        g = 0
        state = env.reset()
        action = np.random.choice(env.action_space.n, p=policy[state])
        while True:
            new_state, reward, done, info = env.step(action)
            new_action = np.random.choice(env.action_space.n, p=policy[new_state])
            
            delta = reward + gamma * q[new_state][new_action] - q[state][action]
            q[state][action] = q[state][action] + alpha * delta
            ac = np.argmax(q[state])
            policy[state] = eps / env.action_space.n
            policy[state][ac] = 1 - eps + eps / env.action_space.n
            state = new_state
            action = new_action
            if done:
                break
        

        if (epi + 1) % 10000 == 0:
            print(f'epi: {epi}, g = {g}')
    return policy, q    

def sarsa_lambda(episode_num, env, gamma, eps, lamb):
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    q = np.zeros_like(policy)
    alpha = 0.1
    e = np.zeros_like(q)
    for epi in range(episode_num):
        g = 0
        state = env.reset()
        action = np.random.choice(env.action_space.n, p=policy[state])
        while True:
            new_state, reward, done, info = env.step(action)
            new_action = np.random.choice(env.action_space.n, p=policy[new_state])
            
            delta = reward + gamma * q[new_state][new_action] - q[state][action]
            e[state][action] += 1
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):

                    q[state][action] = q[state][action] + alpha * delta * e[s][a]
                    e[s][a] = lamb * gamma * e[s][a]
            ac = np.argmax(q[state])
            policy[state] = eps / env.action_space.n
            policy[state][ac] = 1 - eps + eps / env.action_space.n
            state = new_state
            action = new_action
            if done:
                break
        
        if (epi + 1) % 10000 == 0:
            print(f'epi: {epi}, g = {g}')
    return policy, q 

if __name__ == '__main__':
    gamma = 0.9
    eps = 0.1
    episode_num = 50000
    lamb = 0.5
    is_lambda = False
    if not is_lambda:
        p, q = sarsa(episode_num, env, gamma, eps)
    else:
        p, q = sarsa_lambda(episode_num, env, gamma, eps, lamb)
    # test

    state = env.reset()
    while True:
        action = p[state].argmax()# np.random.choice(env.action_space.n, p=p[state])
        state, reward, done, _ = env.step(action)
        print(state, reward, done)
        env.render()
        time.sleep(1)
        if done:
            break