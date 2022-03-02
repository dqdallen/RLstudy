import numpy as np

# 定义矩阵中元素的v值和方向
class Element:
	def __init__(self, value, direct):
		self.v = value
		self.d = direct

# 初始化，v都是0，每个方向的概率一样
arr = []
for i in range(4):
	tmp = []
	for j in range(4):
		tmp.append(Element(0., [1, 1, 1, 1]))
	arr.append(tmp)
# 定义gamma和reward
gamma = 1
r = -1
# 策略迭代
def policy_iter(gamma, r, arr):
	pre = [] # pre用于记录第t次计算的时候的arr，用于t+1次计算arr
	# 将arr矩阵中的复制给pre
	for i in range(4):
		tmp = []
		for j in range(4):
			tmp.append(Element(arr[i][j].v, arr[i][j].d.copy()))
		pre.append(tmp)

	policy_stable = False
	itt = 0
	# 如果策略不变或者迭代次数到1000次就结束

	while not policy_stable and itt < 1000:
		itt += 1
		iter = 0
		# 计算所有状态的值，知道收敛，即值不再变化或者变化很小
		# 策略评估
		while True:
			iter += 1
			maxn = 0
			# 遍历所有状态
			for i in range(4):
				for j in range(4):
					# 左上角和右下角为终点，所以不遍历
					if i == j and i == 0:
						continue
					if i == j and i == 3:
						continue
					v = 0
					d_cnt = 0
					# 遍历四个动作，右r, 左l, 下b, 上t
					for ind, x, y in [(0, 0, 1), (1, 0, -1), (2, 1, 0), (3, -1, 0)]:
						# 这个if是根据之前的策略，例如刚开始大家等概率，则都是1，都遍历
						# 经过策略提升后，得到新的策略，有的是0，有的是1
						if pre[i][j].d[ind] == 0:
							continue
						d_cnt += 1
						xx = i + x
						yy = j + y
						if xx < 0 or yy < 0 or xx >= 4 or yy >= 4:
							v += (r + gamma * pre[i][j].v)
						else:
							v += (r + gamma * pre[xx][yy].v)
					arr[i][j].v = 1.0 / d_cnt * v
					maxn = max(maxn, abs(arr[i][j].v - pre[i][j].v))
			# 用pre记录当前arr的元素
			for i in range(4):
				for j in range(4):
					pre[i][j].v = arr[i][j].v
			if maxn < 0.01:
				break
			# if iter % 10 == 0:
			# 	break
		cnt = 0
		# 策略提升
		for i in range(4):
			for j in range(4):
				# 终点不管
				if i == j and (i == 0 or i == 3):
					continue
				# 记录四个动作对应的v
				a_arr = [float('-inf')] * 4
				for ind, x, y in [(0, 0, 1), (1, 0, -1), (2, 1, 0), (3, -1, 0)]:
					if arr[i][j].d[ind] == 0:
						continue
					xx = i + x
					yy = j + y
					if xx < 0 or yy < 0 or xx >= 4 or yy >= 4:
						a_arr[ind] = (r + gamma * arr[i][j].v)
					else:
						a_arr[ind] = (r + gamma * arr[xx][yy].v)
				# 选择v最大的动作
				max_va = max(a_arr)
				for k in range(4):
					# 不是最大v对应的动作后续不进行考虑，所以置0
					if a_arr[k] != max_va:
						arr[i][j].d[k] = 0
		# 用于计算策略是否平稳，即策略是否变化
		# 如果不变了就不用再循环了
		for i in range(4):
			for j in range(4):
				for k in range(4):
					if arr[i][j].d[k] == pre[i][j].d[k]:
						cnt += 1

		if cnt == 64:
			policy_stable = True
			print(policy_stable, itt)
			action=['右', '左', '下', '上']
			for i in range(4):
				for j in range(4):
					ac = []
					for k in range(4):
						if arr[i][j].d[k] == 1:
							ac.append(action[k])
					print(f'状态({i}, {j}), 方向{arr[i][j].d}, {ac}')

					# print(arr[i][j].d) # r, l, b, t
		else:
			for i in range(4):
				for j in range(4):
					pre[i][j].d = arr[i][j].d.copy() 

# 值迭代和策略迭代类似
# 区别在于策略迭代是要等值函数收敛后，在进行策略提升
# 值迭代是值函数每次所有状态更新后，找到v最大的作为新的值，并保存对应的动作
# 到最后v不变后就直接用保留的action
def value_iter(gamma, r, arr):
	pre = []
	for i in range(4):
		tmp = []
		for j in range(4):
			tmp.append(Element(arr[i][j].v, arr[i][j].d.copy()))
		pre.append(tmp)
	
	iter = 0
	while iter < 10000:
		iter += 1
		maxn = 0
		cnt = 0
		for i in range(4):
			for j in range(4):
				if i == j and i == 0:
					continue
				if i == j and i == 3:
					continue
				v = 0
				a_arr = [float('-inf')] * 4
				for ind, x, y in [(0, 0, 1), (1, 0, -1), (2, 1, 0), (3, -1, 0)]:
					if pre[i][j].d[ind] == 0:
						continue
					xx = i + x
					yy = j + y
					if xx < 0 or yy < 0 or xx >= 4 or yy >= 4:
						a_arr[ind] = r + gamma * pre[i][j].v
					else:
						a_arr[ind] = r + gamma * pre[xx][yy].v
				max_va = max(a_arr)
				arr[i][j].v = max_va
				for k in range(4):
					if a_arr[k] != max_va:
						arr[i][j].d[k] = 0

				maxn = max(maxn, abs(arr[i][j].v - pre[i][j].v))
		for i in range(4):
			for j in range(4):
				pre[i][j].v = arr[i][j].v
		if maxn < 0.01:
			print(iter)
			break

	
	action=['右', '左', '下', '上']
	for i in range(4):
		for j in range(4):
			ac = []
			for k in range(4):
				if arr[i][j].d[k] == 1:
					ac.append(action[k])
			print(f'状态({i}, {j}), 方向{arr[i][j].d}, {ac}')
			# print(arr[i][j].d) # r, l, b, t


# value_iter(gamma, r, arr)			
policy_iter(gamma, r, arr)
