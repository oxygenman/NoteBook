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

![img](/home/xy/pan/xy_workspace/git_workspace/notebook/facerecognition/softmax-based%20loss%20survey.assets/v2-7cb3f114d51b6a129d0075c06c9cb20b_720w.jpg)

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

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D++%3D%5Csum_%7Bi%3D1%2Ci%5Cneq+y%7D%5E%7BC%7D+%5Cmax%28+z_i+-+z_y%2C+0%29) 。

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

这里出现了一个经典的歧义，softmax实际上并不是max函数的smooth版，而是one-hot向量（最大值为1，其他为0）的smooth版。其实从输出上来看也很明显，softmax的输出是个向量，而max函数的输出是一个数值，不可能直接用softmax来取代max。max函数真正的smooth版本是LogSumExp函数（[LogSumExp - Wikipedia](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/LogSumExp)），对此感兴趣的读者还可以看看这个博客：[寻求一个光滑的最大值函数 - 科学空间|Scientific Spaces](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/3290)。

使用LogSumExp函数取代max函数：

$\mathcal{L}_{l s e}=\max \left(\log \left(\sum_{i=1, i \neq y}^{C} e^{z_{i}}\right)-z_{y}, 0\right)$

LogSumExp函数的导数恰好为softmax函数：

$\frac{\partial \log \left(\sum_{i=1, i \neq y}^{C} e^{z_{i}}\right)}{\partial z_{j}}=\frac{e^{z_{j}}}{\sum_{i=1, i \neq y}^{c} e^{z_{i}}} \circ$

经过这一变换，给予非目标分数的1的梯度将会通过LogSumExp函数传播给所有的非目标分数，各个非目标分数得到的梯度是通过softmax函数进行分配的，较大的非目标分数会得到更大的梯度使其更快地下降。这些非目标分数的梯度总和为1，目标分数得到的梯度为-1，总和为0，绝对值和为2，这样我们就有效地限制住了梯度的总幅度。

LogSumExp函数值是大于等于max函数值的，而且等于取到的条件也是非常苛刻的（具体情况还是得看我的博士论文，这里公式已经很多了，再写就没法看了），所以使用LogSumExp函数相当于变相地加了一定的 m。但这往往还是不够的，我们可以选择跟hinge loss一样添加一个 ![[公式]](https://www.zhihu.com/equation?tex=m) ，那样效果应该也会不错，不过softmax交叉熵损失走的是另一条路：继续smooth。

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