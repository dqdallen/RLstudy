import gym
import numpy as np
import time
import pickle


env = gym.make("Taxi-v2") # 不同版本可以使用taxi-v3或者taxi-v2
state = env.reset()
p, q = pickle.load(open('policy_q_sarsa.pickle', 'rb'))
while True:
    action = p[state].argmax()
    state, reward, done, _ = env.step(action)
    print(state, reward, done)
    env.render()
    time.sleep(1)
    if done:
        break