# 问题
这是gym中的经典问题CartPole-v1，通过施加左右方向的力使得棍子能够尽量久的保持竖直。可点击下面的地址前往

http://gym.openai.com/envs/CartPole-v1/
# 代码
* model.py包含DQN模型部分和损失函数
* DQN.py包含训练部分的代码
* test.py包含测试部分的代码
* dqn_catpole.pth存储了训练好了DQN参数，可利用dqn.load_state_dict(torch.load('dqn_catpole.pth'))
# 结果
在gym环境中执行测试后的结果如下所示，大家可以尝试一下

![结果](https://github.com/dqdallen/RLstudy/blob/main/DQN/result.gif)
