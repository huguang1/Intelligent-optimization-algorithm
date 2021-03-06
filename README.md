# 智能优化算法调研
 
### 1	智能优化算法介绍
##### 1.1	智能优化算法概念
优化算法是一种根据概率按照固定步骤寻求问题的最优解的过程。与常见的排序算法、寻路算法不同的是，优化算法不具备等幂性，是一种概率算法。<br>
##### 1.2	智能优化算法的作用
寻找问题最优解。<br>
##### 1.3	智能优化算法特点
###### 1.3.1	启发式智能算法特点
（1）都是一类不确定算法。不确定性体现了自然界生物的生物机制，并且在求解某些特定问题方面优于确定性算法。仿生优化算法的不确定性是伴随其随机性而来的，其主要步骤含有随机因素，从而在算法的迭代过程中，事件发生与否有很大的不确定性。<br>
（2）都是一类概率型的全局优化算法。非确定算法的优点在于算法能有更多机会求解全局最优解。<br>
（3）都不依赖于优化问题本身的严格数学性质。在优化过程中都不依赖于优化问题本身严格数学性质（如连续性，可导性）以及目标函数和约束条件精确的数学描述。<br>
（4）都是一种基于多个智能体的仿生优化算法。仿生优化算法中的各个智能体之间通过相互协作来更好的适应环境，表现出与环境交互的能力。<br>
（5）都具有本质并行性。仿生优化算法的本质并行性表现在两个方面：仿生优化计算的内在并行性（inherent parallelism ）和内含并行性（implicit parallelism ），这使得仿生优化算法能以较少的计算获得较大的收益。<br>
（6）都具有突出性。仿生优化算法总目标的完成是在多个智能体个体行为的运动过程中突现出来的。<br>
（7）都具有自组织和进化性。具有记忆功能，所有粒子都保存优解的相关知识。在不确定的复杂时代环境中，仿生优化算法可以通过自学习不断提高算法中个体的适应性。<br>
（8）都具有稳健性。仿生优化算法的稳健性是指在不同条件和环境下算法的实用性和有效性。由于仿生优化算法不依赖于优化问题本身严格数学性质和所求问题本身的结构特征，因此用仿生优化算法求解不同问题时，只需要设计相应的评价函数（代价函数），而基本上无需修改算法的其他部分。但是对高维问题复杂问题，往往会遇到早熟收敛和收敛性能差的缺点，都无法保证收敛到最优点。<br>
### 2	优化算法技术
##### 2.1	总体分类架构图
常见的分支有群智能和进化两个方向。<br>
##### 2.2	群体智能算法
###### 2.2.1	群体智能发展进程
###### 2.2.2	群体智能经典算法
1. 蚁群优化（ACO）, 用于在图中寻找最优路径。蚁群优化算法的灵感来自蚂蚁寻找巢穴与食物源之间最短路径的能力	否。启发因子。	优点,分布式性、鲁棒性强，动态性、随机性和异步性，全局搜索能力强。缺点, 算法性能跟参数有关，参数由实验确定，与人的经验相关，算法性能难以优化。收敛慢，陷入局部最优。离散问题应用广泛。<br>
2. 粒子群优化算法(PSO),	Vi=Vi+c1*rand()*(Pbesti−Xi)+c2*rand()*(Gbest−Xi)Vi=Vi+c1r1(Pbesti−Xi)+c2r2(Gbest−Xi) （速度更新公式）Xi=Xi+Vi(距离更新公式)。优点,容易、精度高、收敛快，可并行。参数容易优化。全局搜索能力强。缺点,出现早熟收敛而导致优化性能降低。收敛较慢。算法不稳定。局部寻优能力差。	<br>
3. 人工鱼群算法（AFSA）, 在一片水域中，鱼存在的数目最多的地方就是本水域富含营养物质最多的地方，依据这一特点来模仿鱼群的觅食、聚群、追尾等行为，从而实现全局最优。优点,目标函数简单，容错好，并行，全局搜索能力强，应用范围广。	缺点,收敛慢。<br>
4. 细菌觅食优化（BFOA）, 细菌觅食算法是基于细菌觅食行为过程而提出的一种仿生随机搜索算法．该算法模拟细菌群体的行为，包括趋化，繁殖，驱散等三个个步骤。优点，简单、收敛速度快。缺点, 高维多模态问题收敛差。应用广泛。<br>
5. 蛙跳（SFLA）, 模拟石块上觅食时的种群分布变化算法。湿地内离散的分布着许多石头，青蛙通过寻找不同的石头进行跳跃去找到食物较多的地方	全局搜索能力和局部搜索能力相当。不是黑盒算法。优点,算法效果稳定。流程，算子简单。参数少，全局寻优能力强。善于解决多目标问题。缺点,全局搜索能力差，应用不够成熟。<br>
6. 人工蜂群(ABC),	模拟蜂群通过个体分工和信息交流，相互协作完成采蜜任务的算法。不需要了解问题的特殊信息，只需要对问题进行优劣的比较，通过各人工蜂个体的局部寻优行为	优点, 模拟结果比PSO,EA,DE效果好，高效解决多维度问题。简单，适合解决连续或近似连续的问题。参数少，计算简洁，易于实现，鲁棒性强。解决高纬度问题。缺点, 搜索效率低，早熟收敛，收敛速度慢。	应用广泛。<br>
7. 猫群算法（CSO）, 猫群算法正是通过将猫的搜寻和跟踪两种行为结合起来，提出的一种解决复杂优化问题的方法。是遗传算法和粒子群算法的简化版混合。优点, 全局优化算法具有收敛快，寻优能力强的特点，缺点, 早熟收敛，思想、原理、参数设置以及种群多样性的研究, 仍停留在实验探索阶段, 并未有更深入的分析与讨论。应用较少。<br>
8. 萤火虫群优化（GSO）, 在群体中，每个萤火虫个体被随机分布在目标函数定义的空间中，初始阶段，所有的萤火虫都具有相同的荧光素值和动态决策半径。其中，每个萤火虫个体根据来自动态决策半径内所有邻居萤火虫信号的强弱来决定其移动的方向。萤火虫的动态决策半径会随着在它范围内萤火虫个体的数目而变化，每个萤火虫的荧光素也会随着决策半径内萤火虫个体的数目而改变。优点, 速度快，调节参数少，易于实现等	缺点, 局部搜索能力强。有良好应用前景。<br>
9. 蟑螂群优化（CSO）, CSO算法是通过模仿蟑螂的生物学行为来实现的：聚集、分散和残忍行为。优点, 。缺点, 初始解质量不高和算法评价次数过多的问题。应用较少。<br>
10. 布谷鸟搜索（CS-2009）, 布谷鸟搜索受布谷鸟的巢寄生行为和一些鸟类和果蝇的莱维（Lévy Flight）行为的启发	全局搜索使用 Lévy 飞行，而不是标准的随机行走，局部搜索和全局搜索能力。优点, 分布式性、鲁棒性强，动态性、随机性和异步性，全局搜索能力强。缺点, 收敛速度慢，缺少活力。难以解决离散化问题。	CS算法作为后起之秀，它的优越性使其广泛应用于各个研究领域。<br>
11. 萤火虫算法（FA）,	是一种模仿萤火虫之间信息交流，相互吸引集合，警戒危险。	否	优点,容易、精度高、收敛快，可并行。参数容易优化。全局搜索能力强。	缺点,求解精度不高。后期收敛速度慢	处于初级阶段。<br>
12. 蝙蝠算法（BA）（2010）,	pso改进，蝙蝠算法的每只蝙蝠多了频率属性和响度	否	优点,目标函数简单，容错好，并行，全局搜索能力强，应用范围广。 缺点, 在着后期收敛速度慢、收敛精度不高、易陷入局部极小点等不足。新兴算法，有改进点、创新点及应用点。<br>
13. 烟花算法（FWA-2010）, 受到烟花在夜空中爆炸产生火花并照亮周围区域这一自然现象的启发。优点, 简单、收敛速度快。缺点, 易于陷入局部最优。处于初步研究阶段。<br>
14. 蜘蛛猴优化（SMO）,	蜘蛛猴的觅食行为是基于分裂融合的社会结构。优点, 算法效果稳定。流程，算子简单。参数少，全局寻优能力强。善于解决多目标问题。缺点,性能非常依赖于参数选择。应用广泛。<br>
##### 2.3	进化算法
###### 2.3.1	进化算法发展进程
遗传算法的发展历程，大致分为三个时期：萌芽、成长和发展。<br>
萌芽期（50年代后期~70年代中期）。<br>
早在上世界50年代后期和60年代初期，一些生物学家就已经开始采用电子计算机模拟生物的遗传进化系统，尽管这些工作纯粹服务于研究生物现象，但是他们已经开始使用现代遗传算法的一些标识方式，如1960年，美国A.S.Fraser为了建立生物的表现型方程，用3组5位（共15位）长的0-1字符串表示方程的三个参数。<br>
1962年，美国J.H.Holland教授在研究适应系统时，提出了系统本身与外部环境的相互作用与协调，就已经涉及到了进化算法的思想。之后他又进行了许多适应系统方面的研究工作，在1968年他又提出了模式理论，奠定了遗传算法的主要理论基础。<br>
1967年美国的J.D.Bagay在其关于博弈论的论文中，第一次使用了遗传算法（Genetic algorithm）这个术语，他采用了复制、交换、突变等手段研究轨迹象棋的对弈策略。<br>
1975年，J.H.Holland教授出版了著作《自然界和人工系统适应性（Adaptation in Natural and Artificial Systems）》，比较全面地介绍了遗传算法，人们常把这一事件作为遗传算法得到承认的标志，1975年也就成为了遗传算法的诞生年，Holland也就成为了遗传算法的创始人。<br>
成长期（70年代中期~80年代末）。<br>
从70年代末至80年代初，许多研究工作者开始从事遗传算法的研究，遗传算法一度成为美国人工智能研究的一个热点。<br>
1987年，美国的D.Lawrence总结了人们长期从事遗传算法的经验，公开出版了《遗传算法与模拟退火（Genetic Algorithm and Simulated Annealing）》一书，以论文集形式用大量的实例介绍了遗传算法的使用技术。<br>
1989年J.H.Holland教授的学生，D.E.Goldberg博士出版专著《遗传算法—搜索、优化及机器学习（Genetic Algorithms—in Search，Optimization and Machine Learning）》，非常全面、系统地介绍了遗传算法的基本原理和应用，使得这一优化技术得到普及与推广，该书从而也被视为遗传算法的基础教科书。<br>
至此，遗传算法已广泛应用于生物、工程、运筹学、计算机科学、图像处理、模式识别和社会科学等领域。1985年在美国举办了第一届遗传算法国际学术会议（International Conference on Genetic Algorithms），此后又举办了很多届。<br>
发展期（90年代）。<br>
90年代以后，遗传算法不断地向广度和深度发展，1991年，D.Lawrence公开发行《遗传算法手册（Handbook of Genetic Algorithms）》一书，同时随着应用的广泛，遗传算法也暴露出了在表达方面的局限性，在1989年，美国斯坦福大学的J.R.Koza提出了遗传规划新概念，用层次化的计算机程序代替字符串表达问题，这里不对该概念展开，后面会单独介绍该算法。<br>
###### 2.3.2	进化算法经典算法
遗传规划（GA）,	模拟生物在自然环境中遗传和进化的自适应（对遗传参数的自适应调整）全局优化（随机变异不断寻找全局最优解）算法。优点, 并行，可扩展与其他算法结合。较好的全局搜索能力，收敛快。缺点,  1.局部搜索能力较差，导致单纯的遗传算法比较费时，在进化后期搜索效率较低，2.容易产生早熟收敛的问题。3.随机搜索导致退化问题。应用广泛。<br>
文化基因算法（Memetic Algorithm，MA）, 种群的全局搜索和基于个体的局部启发式搜索的结合体。优点, 搜索效率比普通遗传算法快。应用广泛。<br>
差分进化（DE）,	是遗传算法的的改进。通过对父类2个个体的差值和第3个个体相加得到实验个体。实验个体和父代个体交叉生成新的子代个体。优点, 比遗传算法的逼近效果好。算法稳定。。缺点, 参数选择需要试错。早熟收敛。个体少时，难以收敛。高维问题很难处理。将会热门。<br>
免疫算法（IA）,	更新亲和度（这里对应上面的适应度）的过程，抽取一个抗原（问题），取一个抗体（解）去解决，并计算其亲和度，而后选择样本进行变换操作（免疫处理），借此得到得分更高的解样本，在一次一次的变换过程中逐渐接近最后解	优点, 全局收敛。有自组织、自适应、鲁棒性的特点。缺点, 算法复杂	。应用较成熟，网络的入侵检测技术上，已经有了较为全面的研究。<br>
稻田算法, 受进化和免疫算法启发。越接近优化解的植株产生的种子越多	优点, 全局搜索能力，在同等算法效果情况下，需要的计算量比GA少。	<br>
协方差矩阵自适应CMA-ES，ES的自适应版本，主要基于突变因子分布的自适应机制 实现进化策略参数优化调整。优点, 在中等规模（变量个数大约在 3-300范围内）的复杂优化问题上具有很好的效果，无梯度优化，不使用梯度信息。缺点, 算法复杂。应用 最多性能最好。<br>
##### 2.4	神经网络算法
###### 2.4.1	神经网络经典优化算法
梯度下降（GD），梯度下降的方向求解极小值，也可以沿梯度上升方向求解最大值。优点，波动性较强，凸函数收敛到全局最优。缺点非凸函数无法收敛到局部最优。需要目标函数可导。广泛用于各神经网络。<br>
BFGS，与GD类似，用二阶倒数求解极值。优点，收敛速度快。	缺点，计算复杂，大型数据集上无法使用。<br>
共轭梯度法。解决大规模问题。复杂度O（n），计算简单。	<br>

###### 2.4.2	神经网络算法发展进程
神经网络算法发展历程如下。<br>
 

##### 2.5	禁忌搜索算法
###### 2.5.1	算法介绍
是一种随机搜索算法，它从一个初始可行解出发，选择一系列的特定搜索方向(移动)作为试探，选择实现让特定的目标函数值变化最多的移动。为了避免陷入局部最优解，TS搜索中采用了一种灵活的“记忆”技术，对已经进行的优化过程进行记录和选择，指导下一步的搜索方向。优点，简单，较强通用性。收敛快。缺点，搜索结果完全依赖于初始解和邻域的映射关系，全局开发能力弱，局部开发能力强。<br>
##### 2.6	模拟退火算法
###### 2.6.1	模拟退火算法介绍
是一种随机搜索算法，它从一个初始可行解出发，选择一系列的特定搜索方向(移动)作为试探，选择实现让特定的目标函数值变化最多的移动。为了避免陷入局部最优解，TS搜索中采用了一种灵活的“记忆”技术，对已经进行的优化过程进行记录和选择，指导下一步的搜索方向。优点，简单，较强通用性。收敛快。缺点，搜索结果完全依赖于初始解和邻域的映射关系，全局开发能力弱，局部开发能力强。<br>

##### 2.7	混合算法
###### 2.7.1	
### 3	优化算法实现
##### 3.1	源代码
###### 3.1.1	部分源代码参考
参考地址如下：https://github.com/guofei9987/scikit-opt  <br>
### 4	总结
##### 4.1	算法特性总结
###### 4.1.1	总结
1.	蚁群算法和粒子群算法基本是所有群智能算法的祖先。后面算法基本上都是在搜索策略等方面的改进。<br>
2.	比如遗传算法比较适合二进制，因为它本身的物理意义就是用基因，或者说10这样的数字来表示，而粒子群这样的算法更倾向于模拟量，也就是连续变化的量，这算是一个简单的区分。<br>
###### 4.1.2	常见各算法对比表
 
###### 4.1.3	改进方向
但是现在存在的问题大于优点，很多情况我们只是希望能够实线全局优化，但并不是必须要全局优化，这就导致使用这个方法似乎并不是必要的，另一点是，这类算法普遍没有稳定性证明，并且系统的好坏非常依赖参数设置。比如粒子群算法为例，目前很少有人能够确定下来一个普适性的收敛证明，每个问题里面的三个基础参数值是不同的，参数选择范围没有严格的定义，系统的好坏完全根据人的经验而来，随机性过大。 <br>
### 5	参考         
[1]	https://www.zhihu.com/question/30326374 # 智能优化算法总体介绍<br>
[2]	https://zhuanlan.zhihu.com/p/137408401 # 蚁群算法<br>
[3]	https://www.jianshu.com/p/9ef24ad65191 # 蚁群算法及其应用实例<br>
[4]	http://3ms.huawei.com/km/blogs/details/5793515 # 粒子群优化算法<br>
[5]	https://blog.csdn.net/hba646333407/article/details/103087777 # 群体智能优化算法总结<br>
[6]	https://blog.csdn.net/cccddduil/article/details/124903273 # 人工鱼群算法python实现<br>
[7]	https://zhuanlan.zhihu.com/p/100920122 # 人工鱼群算法 超详细解析<br>
[8]	https://blog.csdn.net/wp_csdn/article/details/54577567 # 人工鱼群算法详解<br>
[9]	https://www.bbsmax.com/A/KE5QE9mk5L/ # 细菌觅食优化算法python实现<br>
[10]	https://blog.csdn.net/hba646333407/article/details/103086793 # 群体智能优化算法之细菌觅食优化算法<br>
[11]	https://blog.csdn.net/xiaobiyin9140/article/details/88085607 # 细菌觅食算法<br>
[12]	https://blog.csdn.net/wh_17426/article/details/108960581 # 人工蜂群算法的python实现<br>
[13]	https://www.cnblogs.com/ybl20000418/p/11366576.html # 人工蜂群算法原理<br>
[14]	https://baike.baidu.com/item/%E7%8C%AB%E7%BE%A4%E7%AE%97%E6%B3%95/19460994 # 猫群算法<br>
[15]	https://blog.csdn.net/qq_40731332/article/details/103589592 # 标准萤火虫算法及Python实现<br>
[16]	https://blog.csdn.net/hba646333407/article/details/104798762 # 群体智能优化算法之萤火虫算法<br>
[17]	https://vlight.me/2017/12/17/Cuckoo-Search/ # 布谷鸟搜索算法<br>
[18]	https://github.com/SJ2050SJ/Optimization_Algorithms # 布谷鸟搜索算法python实现<br>
[19]	https://blog.csdn.net/welcome_yu/article/details/112131446 # 蝙蝠算法python实现<br>
[20]	https://www.jianshu.com/p/1cd814484bb0 # 蝙蝠算法<br>
[21]	https://blog.csdn.net/hba646333407/article/details/103087596 # 群体智能优化算法之烟花算法<br>
[22]	https://blog.csdn.net/hba646333407/article/details/103068144 # 群体智能之蜘蛛猴优化算法<br>



[3]	https://www.sohu.com/a/203707509_465975 # 进化策略,或遗传算法<br>
[4]	https://zhuanlan.zhihu.com/p/272656135 # 优化算法综述<br>
[5]	https://blog.csdn.net/qq_38384924/article/details/120808518 # 优化算法详述<br>
[7]	https://blog.csdn.net/xt_18829518330/article/details/100636932 # 基础优化算法学习<br>
[8]	https://blog.csdn.net/XLcaoyi/article/details/107915110 # 寻优算法概述<br>
[9]	https://blog.csdn.net/qq997843911/article/details/83445318 # 常见优化算法分类及总结<br>
[10]	https://zhuanlan.zhihu.com/p/99575925 # 禁忌搜索算法求解带时间窗的车辆路径问题<br>
[11]	https://zhuanlan.zhihu.com/p/33184423 # 模拟退火算法学习笔记<br>
[16]	https://www.csdn.net/tags/MtjaUgzsOTMxMDgtYmxvZwO0O0OO0O0O.html # 粒子群、遗传、蚁群、模拟退火和鲸鱼算法优缺点比较<br>
[23]	https://www.secrss.com/articles/29401 # 人工免疫系统的研究进展与展望<br>
[24]	https://blog.csdn.net/a1920993165/article/details/121864546 # 文化基因算法<br>
[25]	https://blog.csdn.net/a1920993165/article/details/121864417 # 各类优化算法入门优秀论文总结目录<br>
[26]	https://blog.csdn.net/hba646333407/article/details/108836648 # （CMA-ES源码）协方差自适应进化策略<br>
[27]	https://blog.csdn.net/weixin_39478524/article/details/105149590 # 协方差自适应调整的进化策略（CMA-ES）<br>
[28]	https://zhuanlan.zhihu.com/p/425439560 # 论文中常用的改进群智能优化算法<br>


   
