## Dance Revolution

### 摘要：

在这篇文章中，作者将基于音乐的舞蹈生成看做一个sequence-to-sequence问题，并设计了一个seq2seq的网络结构，并且提出了一个curiculum学习策略来减轻长时间动作序列生成自回归模型的误差累积。

### 方法：

#### 问题定义：

假设我们有一个数据集$\mathcal{D}=\left\{\left(X_{i}, Y_{i}\right)\right\}_{i=1}^{N}$ ,其中$X=\left\{x_{t}\right\}_{t=1}^{n}$,是一个音乐片段的特征集和，$x_t$ 是一个特征向量，是时间t时刻音乐帧的声学特征 ， $Y=\left\{y_{t}\right\}_{t=1}^{n}$ 是舞蹈动作序列，其中$y_t$ 和$x_t$ 是对齐的。 我们的目标是利用数据集$\mathcal{D}$训练一个生成模型$g(X)$ .使我们能够给一个新的音乐输入$X$,可以生成一个新的$Y$ 。

#### 网络结构： 

网络架构如图所示：

![image-20220114112911334](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220114112911334.png)

如图所示，输入音乐序列$X$ 先经过一个音乐编码器生成隐层特征$Z=\left(z_{1}, \ldots, z_{n}\right)\left(z_{i} \in \mathbb{R}^{d_{z}}\right)$,然后一个RNN舞蹈解码器基于$Z$ 生成舞蹈序列$Y=\left(y_{1}, \ldots, y_{n}\right)$ .

(1)Music Encoder

音乐编码器采用transformer结构，同时，为了控制模型复杂度，作者引入了一个local self-attention机制，使得self-attention的感受野不再是整个音乐序列，而是k-nearest neighbors. 论文公式很复杂其实就是一个全连接层，加一个有感受野限制的self-attentio层。

![image-20220114143148616](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220114143148616.png)

（2）舞蹈动作解码器采用RNN结构，有两个原因：1.RNN的链式结构可以很好的捕捉舞蹈动作间的时空依赖。2. 文章提出的学习方法是专门针对RNN这种自回归网络的。

计算过程如下：

$\begin{aligned}
\hat{y}_{i} &=\left[h_{i} ; z_{i}\right] W^{S}+b \\
h_{i} &=\operatorname{RNN}\left(h_{i-1}, \hat{y}_{i-1}\right)
\end{aligned}$

lstm hidden state 和 encoder的输入拼接

先将用RNN生成decoder的hidden state,再将音乐当前帧的特征和rnn lstm的hidden state拼接，经过一层全连接层生成下一帧的动作。

### 动态学习策略

在自然语言处理中，Exposure bias是一个臭名昭著的问题。Exposure bias指的是训练的时候使用的输入是ground truth,推理的时候的输入的却是模型上一步的输出。模型在训练阶段从未有引入生成偏差。在自然语言处理中，我们可以通过采样策略来解决这个问题。在我们的场景下，exposure bias带来的问题就是在几秒钟之内由于偏差的累积就会造成动作的freeze.

所以本文提出的学习策略是：

在训练过程中：将decoder的输入采用这种形式$Y_{i n}=\left\{y_{0}, \hat{y_{1}}, \ldots, \hat{y_{p}}, y_{p+1}, \ldots, y_{p+q}, \hat{y}_{p+q+1}, \ldots\right\}$ , 即前p step使用模型的生成动作，接着q步使用ground truth 监督，后面继续使用模型生成的动作。其中q是固定的，p随着训练进程不断增加$p=f(t) \in\left\{\text { const },\lfloor\lambda t\rfloor,\left\lfloor\lambda t^{2}\right\rfloor,\left\lfloor\lambda e^{t}\right\rfloor\right\}$ .

最后使用的损失函数是$l_1$ loss.

$\ell_{1}=\frac{1}{N} \sum_{i=1}^{N}\left\|g\left(X_{i}\right)-Y_{t g t}^{(i)}\right\|_{1}$ 



### 2D 关键点转为3D关键点

#### 数据预处理

1. 将所有3d关键点的坐标系转化到相机坐标系
2. 使用stacked hourgalss 网络做2d关键点检测

#### 网络结构

![image-20220126163349861](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220126163349861.png)