# -*- coding: utf-8 -*-
"""
@Author  : HuGuang
@Time    : 2022/07/12 21:17
@File    : antTSP.py
@Title   : 蚁群算法解决TSP的Python实现
"""
from matplotlib import pyplot as plt
import numpy as np
import math
import random

"""符号注释：
cities : 数组,记录个城市坐标
n : 整型，城市数量
m : 整型，蚂蚁个数，取城市数目的1.5倍

NC : 迭代计数器
NC_MAX : 最大迭代次数
ALPHA : 常数，α，信息素启发式因子，α 选择[1, 4]比较合适
BETA : 常数，β，期望启发因子，β 选择[3 4 5]比较合适
RHO : 常数，ρ，信息素蒸发系数，ρ 选择[0.1, 0.2, 0.5]比较合适
Q : 常数，信息素强度
Eta(i,j) : 启发因子η，距离的倒数

Best_Path : 各代最佳路线
Best_Path_Len : 各代最佳路线长度
Best_Path_Len_Average : 各代路线的平均长度

Table_Phe = np.ones((n, n))  # n * n 信息素矩阵，初始值都为1
Table_Path = np.zeros((m, n))  # m * n 路径记录表
Distance = np.zeros((n, n))  # n * n 两城市间距离矩阵
"""
# 1.参数初始化
cities = [[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399],
          [3488, 1535], [3326, 1556], [3238, 1229], [4196, 1044],
          [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
          [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678],
          [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
          [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376],
          [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550],
          [2545, 2357], [2778, 2826], [2370, 2975]]  # 城市坐标矩阵
n = len(cities)  # 城市数量
m = int(n * 1.5)  # 蚂蚁数量，取城市数的1.5倍

NC = 0  # 迭代计数器，从0开始，表示第NC+1次迭代
NC_MAX = 100  # 最大迭代次数，取100~500之间
ALPHA = 1  # 常数，α，信息素启发式因子
BETA = 5  # 常数，β，期望启发因子
RHO = 0.1  # 常数，ρ，信息素蒸发系数
Q = 100  # 常数，信息素强度

Best_Path = np.zeros((NC_MAX, n))  # NC_MAX*n矩阵，记录NC_MAX次迭代的所求最佳路径，每条记录表示一次迭代所的最优路径
Best_Path_Len = np.zeros((NC_MAX, 1))  # NC_MAX*1矩阵，记录每次的最佳路径长度
Best_Path_Len[:, :] = np.inf  # 暂时赋值为inf
Best_Path_Len_Average = np.zeros((NC_MAX, 1))  # 最优路径长度平均

Table_Phe = np.ones((n, n))  # n * n 信息素矩阵，初始值都为1
Table_Path = np.zeros((m, n))  # m * n 路径记录表
Distance = np.zeros((n, n))  # n * n 两城市间距离矩阵

# 计算两城市间的距离
for i in range(0, n):
    for j in range(0, n):
        if i != j:
            Distance[i, j] = math.sqrt((cities[j][0] - cities[i][0]) ** 2 + (cities[j][1] - cities[i][1]) ** 2)
        else:
            Distance[i][j] = 1e-4

Eta = 1. / Distance  # 启发因子，取距离的倒数

# 2.迭代寻找最佳路径
while NC < NC_MAX:
    print(f"第{NC + 1}次迭代")
    # 2.1 m只蚂蚁随机选取出生点
    Start = np.zeros((m, 1))  # 蚂蚁随机出生点矩阵，m行。为每一个蚂蚁随机找到了一个城市作为出发点。
    for i in range(0, m):
        Start[i] = random.randint(0, n - 1)  # 生成随机城市，m个
    Table_Path[:, 0] = Start[:, 0]  # 将各蚂蚁放到各自起始城市。这个表将用来维护每只蚂蚁，遍历整个所有城市的顺序。
    # 2.2 m只蚂蚁逐个选择路径
    for i in range(0, m):  # m只蚂蚁循环
        for j in range(1, n):  # 每只蚂蚁对n个城市逐个访问
            Visited = Table_Path[i, 0:j]  # 将该只访问过的城市保存到矩阵Visited中
            Unvisited = np.zeros((1, n - j))  # 创建未访问城市的记录矩阵Unvisited
            P = np.zeros((1, n - j))  # 创建未访问城市的访问概率矩阵P
            count = 0  # 计数器
            # 逐个城市检索，将未访问的城市放到Unvisited表中
            for k in range(0, n):
                if k not in Visited:
                    Unvisited[:, count] = k
                    count += 1
            # 根据公式计算概率
            for z in range(0, Unvisited.size):
                tao_x = Visited[Visited.size - 1]   # 上一个访问的城市
                tao_y = Unvisited[:, z]   # 可能走到的下一个城市
                tao = np.power(Table_Phe[int(tao_x), int(tao_y)], ALPHA)  # 计算此时的信息素，下一个路径的信息素，信息素越大，值越大
                eta_x = Visited[Visited.size - 1]   # 上一个访问的城市
                eta_y = Unvisited[:, z]   # 可能走到的下一个城市
                eta = np.power(Eta[int(eta_x), int(eta_y)], BETA)   # 根据距离来选择路径，距离越小，值越大
                P[0, z] = tao * eta   # 存储距离和信息素的综合因素，信息素越大，距离越小，其值越大。
            P = P / (np.sum(P))  # 生成选择每个城市的概率值
            # 轮盘赌法，按概率选择下一个城市
            Pcum = np.cumsum(P)  # 这个是依次累加的数组，尽管是叠加，其实主要是一两个城市的值是主要的
            Select = np.argwhere(Pcum > random.random())  # 这样的目的是为了保证随机性，这里的代码是最核心的
            to_visit = Unvisited[:, int(Select[0])]    # 选取第一个城市进行访问
            Table_Path[i, j] = to_visit  # 将代访问城市加入Table_Path矩阵，代表该城市被访问

    # 2.3.记录本次迭代最佳路线
    Length = np.zeros((m, 1))  # m*1矩阵，记录每只蚂蚁的路径长度
    for i in range(0, m):  # 计算每只蚂蚁的路径长度
        Route = Table_Path[i, :]  # 在Table_Path表中找到该蚂蚁的路径行，将整行赋给Route
        for j in range(0, n - 1):  # 对路径长度求和
            Length[i] = Length[i] + Distance[int(Route[j]), int(Route[j + 1])]
        Length[i] = Length[i] + Distance[int(Route[0]), int(Route[n - 1])]  # 从最后一个节点回到起点的距离

    if NC == 0:  # 第一次迭代
        min_length = np.min(Length)  # m只蚂蚁路径长度最小值
        min_index = np.argmin(Length)  # 最小值索引
        Best_Path[NC, :] = Table_Path[min_index, :]  # 最优路径表
        Best_Path_Len[NC, :] = min_length  # 最优路径长度为最小值
        Best_Path_Len_Average[NC, :] = np.mean(Length)  # 求平均
        print(f"最短路径：{Best_Path_Len[NC, :]}")
    else:  # 第NC次迭代
        min_length = np.min(Length)
        min_index = np.argmin(Length)
        Best_Path_Len[NC, :] = min(Best_Path_Len[NC - 1], min_length)
        Best_Path_Len_Average[NC, :] = np.mean(Length)
        if Best_Path_Len[NC, :] == min_length:
            Best_Path[NC, :] = Table_Path[min_index, :]
        else:
            Best_Path[NC, :] = Best_Path[NC - 1, :]
        print(f"最短路径：{Best_Path_Len[NC, :]}")

    # 2.4.更新信息素
    Delta_Table_Phe = np.zeros((n, n))   # 信息素变化量矩阵
    for i in range(0, m):
        for j in range(0, n - 1):
            Delta_Table_Phe[int(Table_Path[i, j]), int(Table_Path[i, j + 1])] \
                = Delta_Table_Phe[int(Table_Path[i, j]), int(Table_Path[i, j + 1])] + Q / Length[i, :]  # 距离越短，信息素浓度越高
        Delta_Table_Phe[int(Table_Path[i, n - 1]), int(Table_Path[i, 0])] \
            = Delta_Table_Phe[int(Table_Path[i, n - 1]), int(Table_Path[i, 0])] + Q / Length[i, :]
    Table_Phe = (1 - RHO) * Table_Phe + Delta_Table_Phe   # 原始信息素的衰减加上增量信息素

    # 2.5. 禁忌表清零
    Table_Path = np.zeros((m, n))
    print("*" * 100)
    NC = NC + 1  # 进入下一次迭代

# 3. 迭代结束，结果输出
shortest_length = np.min(Best_Path_Len)  # 在Best_Path_Len中找出最短长度
shortest_index = np.argmin(Best_Path_Len)  # 取得该最短长的的路径在Best_Path_Len中的位置
Shortest_Route = Best_Path[shortest_index, :]  # 根据位置在bast_path中找到该路径
print(f"最短路径：{Shortest_Route+1}")
print(f"最短路径长度：{shortest_length}")

# 4.可视化输出
# 4.1 最短路径可视化
# 创建窗口
fig1 = plt.figure(figsize=(10, 8), dpi=120)  # 这只图片大小 figsize(len,width),dpi设置像素值

# 准备数据
x = []  # 存放各城市横坐标
y = []  # 存放各城市纵坐标
for i in range(0, len(cities)):  # 取值
    route = cities[int(Shortest_Route[i])]
    x.append(route[0])
    y.append(route[1])

# 绘制坐标
plt.xticks(np.arange(0, 5000, 500))
plt.yticks(np.arange(0, 5000, 500))

plt.xlabel("x")
plt.ylabel("y")

plt.title("最优路径图")
c = 0
for i, j in zip(x, y):  # 给各个点标注
    plt.text(i + 0.1, j + 0.3, str(int(Shortest_Route[c] + 1)), ha='center', va='bottom',
             fontsize=10.5)
    c += 1

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决matplotlib不显示中文
plt.grid(alpha=0.2)  # 绘制网格,alpha=0.2表示透明度

# 画图
plt.plot(x, y, color='orange')
plt.scatter(x, y, c='black', marker='o')

end2Start_x = [x[len(x) - 1], x[0]]
end2Start_y = [y[len(y) - 1], y[0]]
plt.plot(end2Start_x, end2Start_y, color='orange')  # 连接终点和起点，形成回路

# 展示
plt.show()

# 4.2 各代最短路径和平均距离可视化
fig2 = plt.figure(figsize=(10, 8), dpi=120)  # 这只图片大小 figsize(len,width),dpi设置像素值
it = range(0, NC_MAX)  # 迭代次数
d = Best_Path_Len.tolist()  # 距离
a_ave = Best_Path_Len_Average.tolist()  # 平均距离
plt.xticks(np.arange(0, NC_MAX, 20))
plt.yticks(np.arange(1.5e+4, 2.2e+4, 1000))
plt.xlabel("迭代次数")
plt.ylabel("距离")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决matplotlib不显示中文
plt.grid(alpha=0.2)  # 绘制网格,alpha=0.2表示透明度
plt.plot(it, d, color='b', ls='-', label="各代最短距离")
plt.plot(it, a_ave, color='y', ls='-', label="各代平均距离")
plt.legend()
plt.show()


"""
一句话总结：每条蚂蚁根据自己的经验走了一遍之后，再吸收别人的经验来决定下一次怎么走。
"""
