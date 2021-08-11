## softmax-based loss survey



[人脸识别中softmax-based loss的演化史](https://zhuanlan.zhihu.com/p/76391405)

### 1.softmax的定义及其作用

$$
\operatorname{Softmax}\left(z_{i}\right)=\frac{e^{z_{i}}}{\sum_{c=1}^{C} e^{z_{c}}}
$$

其中$z_i$为第i个节点的输出值，C为输出节点的个数，即分类的类别个数。通过softmax函数可以将多分类的输出值转换为范围在[0,1]和为1的概率分布。

从一个简单的基于 Softmax Loss 的例子出发。下图描述了基于 softmax loss 做分类问题的流程。输入一个训练样本，倒数第二层的 feature extraction layer 输出 feature x，和最后一层的 classification layer 的类别权重矩阵 ![[公式]](https://www.zhihu.com/equation?tex=%5Crm%7BW%7D%3D%5C%7B++W_%7B1%7D%2CW_%7B2%7D%2C...%2CW_%7B%5Cemph%7BK%7D%7D+%5C%7D) 相乘，得到各类别的分数，再经过 softmax function 得到 normalize 后的类别概率，再得到 cross-entropy loss。

![preview](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-89e8684c268b99f6b69df61158e5aed0_r.jpg)

类别 weight ![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BW%7D_%7Bk%7D) 可看作是一个类别所有样本的代表。 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bi%7D%5Ek) 是样本 feature 和类别 weight 的点积，可以认为是样本和类别的相似度或者分数。通常这个分数被称为 logit。

Softmax 能够放大微小的类别间的 logit 差异，这使得它对这些微小的变化非常敏感，这往往对优化过程非常有利。我们用一个简单的三分类问题以及几个数值的小实验来说明这个问题。假设正确的类别为 1。如下表的情况（d）所示，正确类别的概率才 1/2，并不高。

如果要进一步提高正确类别概率，需要正确类别分数远高于其它类别分数，需要网络对于不同样本（类别）的输出差异巨大。网络要学习到这样的输出很困难。然而，加了 softmax 操作之后，正确类别概率轻松变为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Be%5E%7B10%7D%7D%7Be%5E%7B10%7D%2Be%5E%7B5%7D%2Be%5E%7B5%7D%7D%3D98.7%5C%25) ，已经足够好了。

![preview](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-9e3eed7cc789953530401db4a1e53693_r.jpg)

可见，softmax 中的指数操作，可以迅速放大原始的 logit 之间的差异，使得**“正确类别概率接近于 1”**的目标变得简单很多。这种效应可以称为“强者通吃”。

### 2.softmax loss

softmax 与 crossentropy 结合组成了softmax loss.假设softmax的过程如下图所示：

![v2-7cb3f114d51b6a129d0075c06c9cb20b_720w](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-7cb3f114d51b6a129d0075c06c9cb20b_720w.jpg)

符号定义如下：

1.输入为**z**向量，$z=[z_1,z_2,z_3,..z_n]$,维度为（1，n）

2.经过softmax函数，$a_{i}=\frac{e^{z_{i}}}{\sum_{k=1}^{n} e^{z_{k}}}$

可得输出*a*向量，$a=\left[\frac{e^{z_{1}}}{\Sigma_{k=1}^{n} e^{z_{k}}}, \frac{e^{z_{2}}}{\Sigma_{k=1}^{n} e^{z_{k}}}, \ldots, \frac{e^{z_{n}}}{\Sigma_{k=1}^{n} e^{z_{k}}}\right]$,维度为（1，n）

3.softmax loss 损失函数定义为L,$L=-\Sigma_{i=1}^{n} y_{i} \ln \left(a_{i}\right)$,L是一个标量，维度为（1,1）

其中y向量为模型的label,维度也是（1，n）,为已知量，一般为onehot形式。

我们假设第J个类别是正确的，则y = [0,0,---,1,..0],只有$y_j = 1$,其余$y_i = 0$ 

那么$L=-y_{j} \ln \left(a_{j}\right)=-\ln \left(a_{j}\right)$

我们的目标是求标量 L 对向量z的导数 $\frac{\partial L}{\partial \boldsymbol{z}}$.

由链式法则，$\frac{\partial L}{\partial \boldsymbol{z}}=\frac{\partial L}{\partial \boldsymbol{a}} * \frac{\partial \boldsymbol{a}}{\partial \boldsymbol{z}}$ 其中a和z均为维度为（1，n）的向量。



1.求$\frac{\partial L}{\partial \boldsymbol{a}}$

由$L=-y_{j} \ln \left(a_{j}\right)=-\ln \left(a_{j}\right)$,可知最终的loss只跟$a_j$有关。

$\frac{\partial L}{\partial \boldsymbol{a}}=\left[0,0,0, \ldots,-\frac{1}{a_{j}}, \ldots, 0\right]$

2.求$\frac{\partial \boldsymbol{a}}{\partial \boldsymbol{z}}$

a 是一个向量，z也是一个向量，则$\frac{\partial \boldsymbol{a}}{\partial \boldsymbol{z}}$ 是一个jacobian矩阵，类似这样：

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-ec8b04ca0ee923686c9f36277ff8b7b1_720w.jpg" alt="img" style="zoom: 80%;" />

可以发现其实jacobian矩阵的每一行对应$\frac{\partial a_{i}}{\partial \boldsymbol{z}}$ .

由于$\frac{\partial L}{\partial \boldsymbol{a}}$ 只有第j列不为0，由矩阵乘法，其实我们只要求$\frac{\partial \boldsymbol{a}}{\partial \boldsymbol{z}}$ 的第j行，也即 $\frac{\partial a_{j}}{\partial \boldsymbol{z}}$ ,

$\frac{\partial L}{\partial \boldsymbol{z}}=-\frac{1}{a_{j}} * \frac{\partial a_{j}}{\partial \boldsymbol{z}}, \text { 其中 } a_{j}=\frac{e^{z_{j}}}{\Sigma_{k=1}^{n} e^{z_{k}}}$ 。

（1）当$i \neq j$ 

$\begin{array}{l}
\frac{\partial a_{j}}{\partial z_{i}}=\frac{0-e^{z_{j}} e^{z_{i}}}{\left(\sum_{k}^{n} e^{z_{k}}\right)^{2}}=-a_{j} a_{i} \\
\frac{\partial L}{\partial z_{i}}=-a_{j} a_{i} *-\frac{1}{a_{j}}=a_{i}
\end{array}$ 

(2)当$i=j$ 

$\begin{array}{l}
\frac{\partial a_{j}}{\partial z_{j}}=\frac{e^{z_{j}} \sum_{k}^{n} e^{z_{k}}-e^{z_{j}} e^{z_{j}}}{\left(\sum_{k}^{n} e^{z_{k}}\right)^{2}}=a_{j}-a_{j}^{2} \\
\frac{\partial L}{\partial z_{j}}=\left(a_{j}-a_{j}^{2}\right) *-\frac{1}{a_{j}}=a_{j}-1 \\
\text { 所以, } \quad \frac{\partial L}{\partial \boldsymbol{z}}=\left[a_{1}, a_{2}, \ldots, a_{j}-1, \ldots a_{n}\right]=\boldsymbol{a}-\boldsymbol{y} \circ
\end{array}$

softmax cross entropy loss的求导结果非常优雅，就等于预测值与one hot label的差。

#### 3.从最优化角度看待softmax损失函数

我们要思考一个问题：使用神经网络进行多分类（假设为C类）的目标函数是什么？

神经网络的作用是学习一个非线性函数f(x),将输入转换成我们希望的输出。这里我们不考虑网络结构，只考虑分类器的haunted，最简单的方法莫过于直接输出一维的类别序号0...C-1。而这个方法的缺点显而易见：我们事先并不知道这些类别之间的关系，而这样做默认 了相近的整数的类是相似的，为什么第2类的左右分别是第1类和第3类，也许2类跟5类更为接近呢？

为了解决这个问题，可以将各个类别的输出独立开来，不再只输出1个数而是输出C个分数，每个类别占据一个维度，这样就没有谁与谁更接近的问题了。那么如果让一个样本的真值标签所对应的分数比其他分数更大，就可以通过比较这C个分数的大小来判断样本的类别了。。这里沿用我的论文[2]使用的名词，称真值标签对应的类别分数为目标分数(target score)，其他的叫非目标分数(non-target score)。

这样我们就得到了一个优化目标：

> 输出C个分数，使目标分数比非目标分数更大。

换成数学描述，设 ![[公式]](https://www.zhihu.com/equation?tex=z%3Df%28x%29%5Cin+%5Cmathcal%7BR%7D%5EC) 、![[公式]](https://www.zhihu.com/equation?tex=y) 为真值标签的序号，那优化目标即为：

> ![[公式]](https://www.zhihu.com/equation?tex=%5Cforall+j+%5Cneq+y%2C%5C+z_y+%3E++z_j) 。

得到了目标函数之后，就要考虑优化问题了。我们可以给 $z_y$一个负的梯度，给其他所有$z_j$ 一个正的梯度，经过梯度下降法，即可使 ![[公式]](https://www.zhihu.com/equation?tex=z_y) 升高而 ![[公式]](https://www.zhihu.com/equation?tex=z_j) 下降。为了控制整个神经网络的幅度，不可以让 ![[公式]](https://www.zhihu.com/equation?tex=z) 无限地上升或下降，所以我们利用max函数，让在 ![[公式]](https://www.zhihu.com/equation?tex=z_y) 刚刚超过 ![[公式]](https://www.zhihu.com/equation?tex=z_j) 时就停止上升：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D++%3D%5Csum_%7Bi%3D1%2Ci%5Cneq+y%7D%5E%7BC%7D+%5Cmax%28+z_i+-+z_y%2C+0%29) 。(当$z_i>z_y$ 时才有损失函数，当$z_y$ 超过$z_i$ 时 梯度变为0)

然而这样做往往会使模型的泛化性能较差，我们在训练集上才刚刚让$z_y$ 超过 $z_j$ ，那测试集很可能就不会超过。借鉴SVM里间隔的概念，我们添加一个参数，让$z_y$ 比 $z_j$ 大过一定的数值才停止：

$\mathcal{L}_{\text {hinge }}=\sum_{i=1, i \neq y}^{C} \max \left(z_{i}-z_{y}+m, 0\right) \text { 。 }$

这样我们就推导出了hinge loss.

为什么hinge loss 在SVM时代大放异彩，但在神经网络时代就不好用了呢？主要就是因为SVM时代我们用的是二分类，通过一些小技巧，比如1vs1,1vsn等方式来做多分类问题。但是如果把hinge loss应用在多分类上的话，当类别数C特别大时，会有大量的非目标分数得到优化，这样每次优化时的梯度幅度不等且非常巨大，极易梯度爆炸。

其实要解决这个梯度爆炸的问题也不难，我们把优化目标换一种说法：

> 输出C个分数，使目标分数比**最大的**非目标分数更大。

跟之前相比，多了一个限制词“最大的”，但其实我们的目标并没有改变，“目标分数比最大的非目标分数更大”实际上等价于“目标分数比所有非目标分数更大”。这样我们的损失函数就变成了：

$\mathcal{L}=\max \left(\max _{i \neq y}\left\{z_{i}\right\}-z_{y}, 0\right)_{\circ}$

在优化这个损失函数的时候，每次最多只会有一个+1的梯度和一个-1的梯度进入网络，梯度幅度得到了限制。但是这样修改每次优化的分数过少，会使得网络的收敛极其缓慢，这时又要祭出smooth大法了。

那么max函数的smooth版是什么？

这里出现了一个经典的歧义，softmax实际上并不是max函数的smooth版，而是one-hot向量（最大值为1，其他为0）的smooth版（感觉这里说法不妥，应该是$\frac{z_{j}}{\sum_{i=1, i \neq y}^{c} z_{i}}$ 的smooth版）。其实从输出上来看也很明显，softmax的输出是个向量，而max函数的输出是一个数值，不可能直接用softmax来取代max。max函数真正的smooth版本是LogSumExp函数（[LogSumExp - Wikipedia](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/LogSumExp)），对此感兴趣的读者还可以看看这个博客：[寻求一个光滑的最大值函数 - 科学空间|Scientific Spaces](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/3290)。

使用LogSumExp函数取代max函数：

$\mathcal{L}_{l s e}=\max \left(\log \left(\sum_{i=1, i \neq y}^{C} e^{z_{i}}\right)-z_{y}, 0\right)$

LogSumExp函数的导数恰好为softmax函数：

$\frac{\partial \log \left(\sum_{i=1, i \neq y}^{C} e^{z_{i}}\right)}{\partial z_{j}}=\frac{e^{z_{j}}}{\sum_{i=1, i \neq y}^{c} e^{z_{i}}} \circ$

经过这一变换，给予非目标分数的1的梯度将会通过LogSumExp函数传播给所有的非目标分数，各个非目标分数得到的梯度是通过softmax函数进行分配的，较大的非目标分数会得到更大的梯度使其更快地下降。这些非目标分数的梯度总和为1，目标分数得到的梯度为-1，总和为0，绝对值和为2，这样我们就有效地限制住了梯度的总幅度。

LogSumExp函数值是大于等于max函数值的，而且等于取到的条件也是非常苛刻的（具体看wangfeng大神的博士论文），所以使用LogSumExp函数相当于变相地加了一定的 m。但这往往还是不够的，我们可以选择跟hinge loss一样添加一个 ![[公式]](https://www.zhihu.com/equation?tex=m) ，那样效果应该也会不错，不过softmax交叉熵损失走的是另一条路：继续smooth。

注意到ReLU函数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmax%28x%2C0%29) 也有一个smooth版，即softplus函数 ![[公式]](https://www.zhihu.com/equation?tex=%5Clog%281%2Be%5Ex%29) 。使用softplus函数之后，即使 ![[公式]](https://www.zhihu.com/equation?tex=z_y) 超过了LogSumExp函数，仍会得到一点点梯度让 ![[公式]](https://www.zhihu.com/equation?tex=z_y) 继续上升，这样其实也是变相地又增加了一点 m，使得泛化性能有了一定的保障。替换之后就可以得到：
$$
\begin{aligned}
\mathcal{L}_{\text {softmax }} &=\log \left(1+e^{\log \left(\sum_{i=1, i \neq y}^{C} e^{z_{i}}\right)-z_{y}}\right) \\
&=\log \left(1+\frac{\sum_{i=1, i \neq y}^{C} e^{z_{i}}}{e^{z_{y}}}\right) \\
&=\log \frac{\sum_{i=1}^{C} e^{z_{i}}}{e^{z_{y}}} \\
&=-\log \frac{e^{z_{y}}}{\sum_{i=1}^{C} e^{z_{i}}}
\end{aligned}
$$
这个就是大家所熟知的softmax交叉熵损失函数了。在经过两步smooth化之后，我们将一个难以收敛的函数逐步改造成了softmax交叉熵损失函数，解决了原始的目标函数难以优化的问题。从这个推导过程中我们可以看出smooth化不仅可以让优化更畅通，而且还变相地在类间引入了一定的间隔，从而提升了泛化性能。

#### softmax 理解之smooth程度控制

先看一组例子：

x = [1 2 3 4]
softmax(x) = [0.0321 0.0871 0.2369 0.6439]
LSE(x) = 4.4402

(LSE=LogSumExp)

从这个例子看，本来1和4只差4倍，通过指数函数的放大作用，softmax后的结果相差大约20倍。这样看，softmax起到了近似one-hot max的作用，但0.6439其实也不算靠近1，近似效果不佳。

下面我们将x放大10倍：

x = [10 20 30 40]
softmax(x) = [9.36e-14 2.06e-9 4.54e-5 1.00]
LSE(x) = 40.00

那缩小10倍呢？

> x = [0.1 0.2 0.3 0.4]
> softmax(x) = [0.2138 0.2363 0.2612 0.2887]
> LSE(x) = 1.6425

可以看到在这种情况下，Softmax 和 LogSumExp 都不再能够很好地近似了，Softmax 的最大值和最小值差距不到一倍，而 LogSumExp 竟然比 max(x) 大了4倍多。

也就是说 Softmax 交叉熵损失在输入的分数较小的情况下，并不能很好地近似我们的目标函数。在使用 ![[公式]](https://www.zhihu.com/equation?tex=LSE%28%5Cmathbf%7Bz%7D%29) 对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmax+%5C%7B+%5Cmathbf%7Bz%7D+%5C%7D) 进行替换时， ![[公式]](https://www.zhihu.com/equation?tex=LSE%28%5Cmathbf%7Bz%7D%29) 将会远大于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmax+%5C%7B+%5Cmathbf%7Bz%7D+%5C%7D) ，不论 ![[公式]](https://www.zhihu.com/equation?tex=z_y) 是不是最大的，都会产生巨大的损失。我们在第一篇文章中分析过，稍高的 ![[公式]](https://www.zhihu.com/equation?tex=LSE%28%5Cmathbf%7Bz%7D%29) 可以在类间引入一定的间隔，从而提升模型的泛化能力。但过大的间隔同样是不利于分类模型的，这样训练出来的模型必然结果不理想。

总结一下， $\boldsymbol{z}$的幅度既不能过大、也不能过小，过小会导致近对目标函数近似效果不佳的问题；过大则会使类间的间隔趋近于0，影响泛化性能。

一种方案出现了:

将特征和权重全部归一化，这样 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D) 就从特征与权重的内积变成了特征与权重的余弦，由于余弦值的范围是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5B+-1%2C+1%5Cright%5D) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D) 的幅度就大致被定下来了，然后我们再乘上一个尺度因子 ![[公式]](https://www.zhihu.com/equation?tex=s) 来拉长 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D) 的幅度，来保证输入到 Softmax 里的分数能在一个合适的范围：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D+%3D+-%5Clog%5Cleft%28+%7B%5Cfrac%7Be%5E%7Bs%5Ctilde%7Bz_y%7D%7D%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BC%7De%5E%7Bs%5Ctilde%7Bz_i%7D%7D%7D%7D+%5Cright%29+) 

这里的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bz%7D) 表示是由归一化后的特征与权重的内积（即余弦）得到的分数。这个方法的优点在于我们是可以估计出 Softmax 大概需要的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D) 的幅度是多少。

#### Softmax理解之margin

在了解如何引入 margin 之前，我们首先要知道为何要加margin。在SVM时代，margin （以下称作间隔）被认为是模型泛化能力的保证，但在神经网络时代使用的最多的损失函数 Softmax 交叉熵损失中并没有显式地引入间隔项。从第一篇和第三篇文章中我们知道通过 smooth 化，可以在类间引入一定的间隔，而这个间隔与特征幅度和最后一个内积层的权重幅度有关。但这种间隔的大小主要是由网络自行调整，并不是我们人为指定的，网络自行调整的间隔并不一定是最优的。实际上为了减小损失函数的值，margin 为零甚至 margin 为负才是神经网络自认为正确的优化方向。

此外，在人脸认证、图像检索、重识别任务中，是通过特征与特征之间的距离来判断两个样本的相似度的。因此我们一般不使用整个网络模型，而是将最后的分类层丢掉，仅仅使用模型提取特征。例如下图的一个二分类问题：

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-1b251f5ab71ede64edea4ce8475d5b32_720w.jpg)

红色点和绿色点可以被一条分界线很好地分开，作为一个分类模型，这条分界线的表现很不错，识别率100%。但如果要进行特征比对任务，这个模型就远远不够精确了：从图上我们可以看到，分界线两侧的两个点之间的距离（黄色箭头）可能会远小于类内的最大距离（蓝色箭头）。所以使用分类器来训练特征比对任务是远远不够的，我们必须将类间的间隔拉得更大，才能保证“类间距离大于类内距离”这一目标。



下面就来谈谈如何提高间隔。在我们这个系列的[第一篇文章](https://zhuanlan.zhihu.com/p/45014864)中，使用了如下优化目标：

> 输出C个分数，使目标分数比**最大的**非目标分数更大。

其对应的损失函数为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D+%3D+%5Cmax%28+%5Cmax_%7Bi%5Cneq+y%7D%5C%7Bz_i%5C%7D+-+z_y+%2B+m%2C+0%29)

这个损失函数的意义是：

输出C个分数，使目标分数比最大的非目标分数还要大 m。

这里的m由我们手动进行调节，m 越大，则我们会强行要求目标与非目标分数之间拉开更大的差距。但注意到，如果我们不限制分数 $\mathcal{z}$ 的取值范围，那网络会自动优化使得分数 $z$整体取值都非常大，这样会使得我们所设置的 m 相对变小，非常不便。

因此我们需要使用第三篇文章中提到的归一化方法，让分数 $z$由余弦层后接一个尺度因子 s 来得到。这样我们就可以手动控制分数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D) 的取值范围，以方便设置参数 m。

于是损失函数变为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D+%3D+%5Cmax%28+%5Cmax_%7Bi%5Cneq+y%7D%5C%7Bsz_i%5C%7D+-+sz_y+%2B+sm%2C+0%29) ，

这里的 s 看似是一个冗余项，损失函数乘以一个常数并不影响其优化。但因为我们后面要将这个损失函数 smooth 化，所以这里仍然保留着 s 项。接下来使用第一篇文章中的 smooth 技巧，将损失函数化为：

$\begin{aligned}
\mathcal{L} &=\max \left(\max _{i \neq y}\left\{s z_{i}\right\}-s z_{y}+s m, 0\right) \\
& \approx \text { Softplus }\left(L S E\left(s z_{i} ; i \neq y\right)-s\left(z_{y}-m\right)\right) \\
&=-\log \left(\frac{e^{s\left(z_{y}-m\right)}}{e^{s\left(z_{y}-m\right)}+\sum_{i \neq y} e^{s z_{i}}}\right)
\end{aligned}$

因为这里的 m 与分数 $z$ 之间是加减的关系，所以我们称这个损失函数为“带有加性间隔的 Softmax 交叉熵损失函数”

（这里需要将x,和w进行归一化处理，$z_y$ 就是w和x夹角$\theta$的余弦$\cos\theta$）

这个损失函数中有两个参数，一个s 一个m.其中 s 的设置参照第三篇文章，其实只要超过一定的阈值就可以了，一般来说需要设置一个最小值，然后让 s 按照正常的神经网络参数的更新方式来更新。对于 m 的设置目前还需要手动调节，下一篇文章会介绍一些自动设置的方法，但效果还是不如调参师手工调整的好。

在不加间隔，即 $m=0$ 时，不同的 s 会产生不同的边界形状：

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-e1bb74577f5df7e13579b26a4ef38c11_720w.jpg)

s 越小，边界变化会越平缓。这样会使得每个类的点更加向类中心收拢，其实也可以起到一定的增加类间间隔的作用。但根据第三篇文章，过小的 s 会带来近似不准确的缺点，所以通过缩小 s 来增加间隔并不是一个好方法。想要添加类间间隔当然还是直接显式地添加比较好。

在$s=30$ 的基础上添加间隔项 m 的效果如下：

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-d422709c943c40dd027da5c0fde57eb1_b-1626747337955.jpg" alt="v2-d422709c943c40dd027da5c0fde57eb1_b-1626747337955" style="zoom:50%;" />



对比这张图和上边的(c)图，可以看到每个类别所占据的区域减小了，样本只有落在红色区域内才被认为是分类正确的。从这张图上可以看出，由于限制了特征空间在单位球上，添加间隔项 m 可以同时增大类间距离并缩小类内距离。而且这个间隔可以由我们来手动调节，在过拟合严重的数据集上，我们可以增加 m 来使得目标更难，起到约束的作用，从而降低过拟合效应。

[人脸识别loss 发展史](https://zhuanlan.zhihu.com/p/59440497)

#### ArcFace(CVPR2019)

ArcFace认为角度余量比余弦余量更重要，所以对AM-Softmax进行了改进，得到如下loss函数：

将margin加在角度中：

$L_{3}=-\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s\left(\cos \left(\theta_{y_{i}}+m\right)\right)}}{e^{s\left(\cos \left(\theta_{y_{i}}+m\right)\right)}+\sum_{j=1, j \neq y_{i}}^{n} e^{s \cos \theta_{j}}}$

这就是arcface.

几种softmaxloss 变体的分类界面

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-55da7e5782e27a3ea3ab069be4c65cd3_720w.jpg)

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-3be7bc148d655a5cd1a0f116948386e4_720w.jpg)

#### 乘性margin的影响

但引入 margin 之后，有一个很大的问题，网络的训练变得非常非常困难。在 *[SphereFace]* 中提到需要组合退火策略等极其繁琐的训练技巧。这导致这种加 margin 的方式极其不实用。而事实上，这一切的困难，都是因为引入的 margin 是乘性 margin 造成的。我们来分析一下，乘性 margin 到底带来的麻烦是什么：

1. 第一点，乘性 margin 把 cos 函数的单调区间压小了，导致优化困难。对 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccos%28%5Ctheta_%7By_%7Bi%7D%2Ci%7D%29) ，在 $\theta_{y_i,i}$ 处在区间 [0,π] 时，是一个单调函数，也就是说 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7By_%7Bi%7D%2Ci%7D) 落在这个区间里面的任何一个位置，网络都会朝着把 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7By_%7Bi%7D%2Ci%7D) 减小的方向优化。但加上乘性 margin m 后 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccos%28%5Ctheta_%7By_%7Bi%7D%2Ci%7D%29) 的单调区间被压缩到了 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C%5Cfrac%7B%5Cpi%7D%7Bm%7D%5D) ，那如果恰巧有一个 sample 的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7By_%7Bi%7D%2Ci%7D) 落在了这个单调区间外，那网络就很难优化了；
2. 第二点，乘性 margin 所造成的 margin 实际上是不均匀的，依赖于 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctextbf%7BW%7D_%7B1%282%29%7D) 的夹角。前面我们已经分析了，两个 class 之间的 angular decision margin ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bm-1%7D%7Bm%2B1%7D%5Ctheta_%7B1%2C2%7D) ，其中 $\theta_{1,2}$是两个 class 的 weight 的夹角。这自然带来一个问题，如果这两个 class 本身挨得很近，那么他们的 margin 就小。特别是两个难以区分的 class，可能它们的 weight 挨得特别近，也就是 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7B1%2C2%7D) 几乎接近 0，那么按照乘性 margin 的方式，计算出的两个类别的间隔也是接近 0 的。换言之，乘性 margin 对易于混淆的 class 不具有可分性。