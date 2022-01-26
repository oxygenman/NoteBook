## 人体姿态估计

[openpose论文理解](https://blog.csdn.net/wwwhp/article/details/88782851)

[SMPL:A Skinned Multi-Person Linear ModeL论文解读](https://blog.csdn.net/JerryZhang__/article/details/103478265)

[匈牙利算法](https://zhuanlan.zhihu.com/p/96229700)

[Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf)

[人体姿态估计综述](https://zhuanlan.zhihu.com/p/331564848)

### openpose理解：

#### 摘要：

本文提出一种实时检测多人2D姿态的方法；采用nonparametric representation:Part Affinity Fields(PAFs),(部分亲和度向量场)去学习将身体部位和对应个体关联；提出组合检测器可以减少推理时间，推出openpose,多人2d姿态检测开源实时系统，包括身体，脚部，手部，和面部关键点检测。

#### 方法：

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20190324213405310.png)

- 图2是本文方法整体的pipeline:

  - 输入$w*h$ 图像a,为每一个人产生2d关键点定位e;

  - 实现CNN预测一组身体部位的2D置信图(b) $S$ ,和一组2d PAFs $L$ 对部位之间的联系进行编码图c;

  - 集合$S=(S_1,S_2,...,S_J)$ 有J个置信图，其中$\mathbf{S}_{j} \in \mathbb{R}^{w \times h}, j \in\{1 \ldots J\}$,一个关键点一个置信图。

  - 集合$L=(L_1,L_2,...,L_C)$ 有C个矢量场，每个肢体一矢量场，其中$\mathbf{L}_{c} \in \mathbb{R}^{w \times h \times 2}, c \in\{1 \ldots C\}$ 

  - 图像位置$L_C$ 是一个编码后的2D vector如图1所示，最后通过贪心推理解析置信图和PAF输出所有人的2D关键点；

  - 总结一下整体流程: a输入图像，b预测关键点置信度&c关键点亲和度向量，关键点解析，人体骨骼搭建(连接关键点)

    ![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20190324213432453.png)

#### 网络结构：



![image-20210712132744853](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712132744853.png)

首先由一个VGG-19网络的前10层，提取出特征图feature maps $F$.

接下来的网络由上面的图示的两部分分别循环堆叠而成，

在前$T_p$ 个stage,循环堆叠产生PAFs的网络，每一个stage由F和上一个stage的输出拼接而成。

![image-20210712141316068](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712141316068.png)

当$\forall T_P <t\leq T_P + T_C$ ,后$T_C-T_P$个stage,循环堆叠产生置信图。

![image-20210712142453139](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712142453139.png)

在stage $T_P$ 输入是$L^{T_P}$ 和$F$, $T_p$之后将$S^{t-1}$ 也拼接起来。

![image-20210712144007076](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712144007076.png)

如图所示，随着sagte的加深产生的PAFs的效果在变好。

#### 损失函数

PAF分支在stage $t_i$ 和 confidence map 在 stage $t_k$的损失函数分别是：

![image-20210712144850709](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712144850709.png)

其中$W$是一个binary mask ,当pixel P的标签缺失时，$W(P)=0$.(这样做是为了避免训练时对true positive的惩罚)。损失函数是按关键点进划分的。整体的损失函数如下：

![image-20210712150517317](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712150517317.png)

#### 置信图

置信图groundtruth的生成：

![image-20210712152934130](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712152934130.png)

$X_{j,k} \in \mathbb{R^2} $ 表示第K个person的第j个关键点的位置。$S_{j,k} ^{*}$ 表示置信图在像素位置P的值。

推理阶段，每个位置的置信度：

![image-20210712160654688](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712160654688.png)

#### 身体部位亲和向量场

用来建立身体部位之间的联系。

亲和向量场ground truth的生成：

![image-20210712170406104](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712170406104.png)

如图所示，$X_{j_1,k}$ 和$X_{j_2,k}$ 是肢体的部位$j_1$ 和$j_2$的ground truth.那么对于某个像素位置P,

![image-20210712170915460](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712170915460.png)

其中$\mathbf{v}=\left(\mathbf{x}_{j_{2}, k}-\mathbf{x}_{j_{1}, k}\right) /\left\|\mathbf{x}_{j_{2}, k}-\mathbf{x}_{j_{1}, k}\right\|_{2}$ .

那么如何判断P是否在肢体上呢？

![image-20210712171423124](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712171423124.png)

其中v是两个关键点的连接方向的单位向量，$\sigma$ 是宽度。$l_{c,k}$是两个关键点的距离

如果一张图片中有多个人，则取一个平均：

![image-20210712171636340](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712171636340.png)

在测试阶段，如何评估身体部位之间的联系，即两个身体部位的关联置信度？

通过一个线段积分来实现：

![image-20210712172839091](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712172839091.png)

其中，![image-20210712172947031](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712172947031.png)

实际的实现过程中，sampling and summing uniformly-spaced values of u.对u进行等间隔采样。我理解就是对每个点求一下两个向量的内积，然后相加。

#### 将PAFs用在多人中：

body part detection candidates $D_J=\{d_j^m: for j \in\{1 ... J\}, m\in\{1 .. N_j\}\}$ ,其中 $N_j$是part j candidates 的数量。$d_j^m\in\mathbb{R^2}$  是body part j的第m个candidate的位置。

定义一个变量$z_{j_1j_2}^{mn}\in\{0,1\}$ 来表示两个节点$d_{j_1}^m$ 和$d_{j_2}^n$ 之间是否连接 

我们的目标是在两个part候选节点的所有链接中找到最优的



二分图最大权匹配（maximum weight matching in a bipartite graph）

![image-20210712193026416](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712193026416.png)

$E_{mn}$是$d_{j_1}^m$ 和 $d_{j_2}^n$ 之间的PAF,(如式11定义) $\mathcal{Z}_C$ 表示类型c的肢体。（14）（15）限制了不存在两条相同类型的边共享一个node.(一个肘部不可能和两个肩部相连)，最终的目标是是使使两个集合中的点两两连接，并且使他们的权值的和最大。

我们可以使用匈牙利算法获得最佳匹配。

[知乎加权匈牙利匹配算法 ](https://zhuanlan.zhihu.com/p/62981901)

![image-20210712204426552](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712204426552.png)

NP问题定义：对于一类问题，我们可能没有一个已知的快速的方法得到问题的答案，但是如果给我们一个candidate answer,我们能在polynominal的时间内验证这个答案。

NP-hard问题定义：NP-hard问题至少和NP问题一样难。

解决全集$\mathcal{Z}$ 是一个NP hard 问题,所以我们添加两个relaxations:

1.使用最少的边获得生成树（fig6.c）

2.将matching problem分解成二分图最大权匹配子问题。

通过这两个relaxations 问题被简化为：

![image-20210712204810710](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210712204810710.png)

[lightweight openpose 代码](https://www.cnblogs.com/darkknightzh/p/12152119.html)