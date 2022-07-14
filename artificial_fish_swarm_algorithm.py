import math
import matplotlib.pyplot as plt
import numpy

my_flag = 1


# my_flag == 0 计算第一个函数最大值
# my_flag == 1 计算第二个函数最大值


class fish:
    def __init__(self, div, number, visual, step, try_time, delta):
        # 初始化
        self.div = div
        self.number = number
        self.visual = visual
        self.step = step
        self.try_time = try_time
        self.delta = delta

    def distance(self, f):
        # 计算self鱼和f鱼之间的距离
        ll = len(self.number)
        res = 0.0
        for i in range(ll):
            res = res + (self.number[i] - f.number[i]) * (self.number[i] - f.number[i])
        return math.sqrt(res)

    def func(self, flag):
        # 计算鱼在当前位置的函数
        ll = len(self.number)
        if flag == 0:
            res = 0
            for i in range(ll):
                res = res + self.number[i] * self.number[i]
            return 500 - res
        else:
            res = 0
            mul = 1
            for i in range(ll):
                res = res + math.fabs(self.number[i])  # 求绝对值的函数
                mul = mul * math.fabs(self.number[i])
            if 500 - (res + mul) > 0:
                print(500 - (res + mul))
            return 500 - (res + mul)

    def prey(self):
        # 捕食操作
        pre = self.func(my_flag)
        for i in range(self.try_time):
            rand = numpy.random.randint(-99, 99, self.div) / 100 * self.visual  # 在视线中往前游了一会。
            for j in range(self.div):
                self.number[j] = self.number[j] + rand[j]
            cur = self.func(my_flag)
            if cur > pre:  # 如果函数值变大了，则说明捕食成功
                # 捕食成功
                # print('原始分数：' + str(pre) + '新分数：' + str(cur) + '捕食成功！！')
                return cur
            else:
                # 捕食失败
                for j in range(self.div):
                    self.number[j] = self.number[j] - rand[j]
        # print("捕食失败！")
        return pre

    def swarm(self, fishes):
        # 聚群行为：向视觉内鱼群中心前进step
        close_swarm = find_close_swarm(fishes, self)
        center_f = center_fish(close_swarm)  # 查找到这个鱼的最中心的鱼，之后判断是否跟随
        n = len(close_swarm) - 1
        if n != 0 and (center_f.func(my_flag) / n > self.delta * self.func(my_flag)):
            # print("聚群运动")
            for i in range(self.div):
                self.number[i] = self.number[i] + self.step * center_f.number[i]
            return self.func(my_flag)
        else:
            # print("随机运动")
            return self.rand()

    def rand(self):
        for i in range(self.div):
            self.number[i] = self.number[i] + self.step * numpy.random.uniform(-1, 1, 1)

    def follow(self, fishes):
        # 追尾行为：向着视觉内鱼群中目标函数值最优的鱼前进step
        close_swarm = find_close_swarm(fishes, self)   # 寻找到视线内部的鱼
        best_f = best_fish(close_swarm)  # 获取到最好的鱼
        n = len(close_swarm) - 1
        if n != 0 and (best_f.func(my_flag) / n > self.delta * self.func(my_flag)):  # 如果最好的鱼的位置比我的好，这条鱼就会跑过去
            # 向前移动
            # print("向前移动")
            for i in range(self.div):
                self.number[i] = self.number[i] + self.step * (best_f.number[i] - self.number[i])
            return self.func(my_flag)
        else:
            # 随机运动
            # print("随机运动")
            return self.rand()  # 否则就会随机运动


def find_close_swarm(fishes, fish_):
    # 在种群fishes中查找fish_视觉范围内的鱼
    # 输入为fishes，是一个list型变量 和一个fish对象
    # 输出为一个fish list
    res = []
    for fi in fishes:
        if fish_.distance(fi) < fish_.visual:
            res.append(fi)
    return res


def center_fish(fishes):
    # 计算当前种群的中心位置，并将其中心位置记为certer_fish以完成聚群操作
    # 输入为fishes，是一个list型变量
    # 输出为一个fish对象
    ll = len(fishes)
    if ll == 0 or ll == 1:
        return None
    res = fish(fishes[0].div, fishes[0].number, fishes[0].visual, fishes[0].step, fishes[0].try_time, fishes[0].delta)
    for i in range(fishes[0].div):
        res.number[i] = 0
    for i in range(ll):
        for j in range(res.div):
            res.number[j] = res.number[j] + fishes[i].number[j]
    return res


def best_fish(fishes):
    # 计算当前种群最优个体的位置，并将其返回用于追尾操作
    # 输入为fishes，是一个list型变量
    # 输出为一个fish对象
    ll = len(fishes)
    if ll == 0 or ll == 1:
        return None
    index = -1
    max = 0
    for i in range(ll):
        if index == -1 or max < fishes[i].func(my_flag):
            index = i
            max = fishes[i].func(my_flag)
    return fishes[index]


def main():  # 主函数
    fishes = []
    div = 3  # xi中i的大小，e.g. div == 3 --> x1, x2, x3
    fish_num = 50  # 鱼群个体数目
    gmax = 100  # 循环最大次数
    tag = 0  # 公告牌
    visual = 1
    step = 0.2
    try_time = 10
    delta = 0.3
    list_of_fishes = []
    # 初始化鱼群
    for i in range(fish_num):
        num = numpy.random.uniform(10, 20, div)
        print(num)
        fi = fish(div, num, visual, step, try_time, delta)
        fishes.append(fi)
    for i in range(fish_num):
        list_of_fishes.append([])
    for g in range(gmax):
        print(g)
        for i in range(fish_num):
            if fishes[i].func(my_flag) > tag:   # 根据鱼的初始值来计算一个函数的目标值
                tag = fishes[i].func(my_flag)
        # print(g, tag)
        if g >= 50:
            for i in range(fish_num):
                list_of_fishes[i].append(fishes[i].func(my_flag))  # 当迭代次数达到50次的时候记录下鱼群的目标值
        for i in range(fish_num):
            if tag == fishes[i].func(my_flag):  # 当鱼群中出现正值
                fishes[i].prey()
                continue
            tmp = numpy.random.randint(0, 3, 1)  # 每条鱼都在随机的巡游，跟随或觅食
            if tmp == 0:
                fishes[i].swarm(fishes)  #
            elif tmp == 1:
                fishes[i].follow(fishes)  # 跟随别的鱼
            else:
                fishes[i].prey()  # 向目标函数挺近
    print(tag)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # print(x)
    for i in range(fish_num):
        # print(list_of_fishes[i][49] < 0.5)
        if math.fabs(list_of_fishes[i][49] - 500) < 20:
            plt.plot(x, list_of_fishes[i], color='orangered', marker='o', linestyle='-', label='A')
        else:
            plt.plot(x, list_of_fishes[i], color='green', marker='*', linestyle=':', label='C')

    plt.ylim(20, 50)
    plt.ylim(-500, 500)
    plt.xlabel("Loop_time")  # X轴标签
    plt.ylabel("Value")  # Y轴标签
    plt.show()


if __name__ == '__main__':
    main()








