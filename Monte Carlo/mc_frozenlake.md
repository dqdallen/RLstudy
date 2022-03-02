# 问题
FrozenLake-v0问题是在一个4 * 4矩阵中，从起始位置S，走到终点G，中间有道路F和陷阱H，希望通过训练使得智能体能从S快速到达G。
# 代码
代码中包含利用蒙特卡洛解决强化学习的相关方法，分别包含on policy和off policy的方法，其中on policy中包含first visit和every visit的判断，可以自行选择。
# 运行
python mc_frozenlake.py
# 结果
迭代后，代码中的test部分对其进行测试，运行一次后可以看到已经可以从S到G顺利到达了。

![image1](https://github.com/dqdallen/RLstudy/blob/main/MDP_P_V/result.png)

PS：可能是由于不稳定的问题，有时候迭代后的policy不适用，如果出现结果不对的小伙伴可以多试几次或者调整eps探索。
