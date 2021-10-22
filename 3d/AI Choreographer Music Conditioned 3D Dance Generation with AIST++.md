

[3] Emre Aksan, Peng Cao, Manuel Kaufmann, and Otmar Hilliges. Attention, please: A spatio-temporal transformer for 3d human motion prediction. arXiv preprint arXiv:2004.08692, 2020. 2, 3, 5, 7

## preliminaries

### sequence to sequence 

语音识别 翻译 语音合成

参考李宏毅老师讲解

### self attention 

self attention 输入长度和输出长度相等，



### 4.Music Conditioned 3D Dance Generation 

输入2秒钟的种子动作：$X=(x_1,...,x_T)$ 和一段音乐序列$Y=(y_1,...,y_T^{\prime})$ ,生成后续的动作序列

$X^{\prime} = (x_{T+1},...,x_{T'}) ; T'>>T$ .





## AI Choreographer: Music Conditioned 3D Dance Generation with AIST++

### 1.摘要

发表于ICCV2021的一篇工作，构建了一个有关音乐和三维舞蹈动作的多模态数据库，以及一个基于全注意力机制的多模态Transformer网络FACT，该网络可以在给定一段音乐和舞蹈动作种子的前提下生成符合该音乐的三维舞蹈动作。

两大贡献：

（1）AIST++数据集：目前规模最大且动作最丰富的3D人体关键点标注数据库。原AIST数据库针对10个流派的街舞，包含10,108,015帧从9个方位拍摄的人体图像，且配有对应的音乐。作者对每帧图像都给出了对应的SMPL、2D关键点和3D关键点标记。

![image-20211019165458679](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211019165458679.png)

（2）Full Attention Cross-Modal Transformer model(FACT),作者为这个任务设计的网络模型

![image-20211019170136485](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211019170136485.png)

### 主要方法（即FACT模型的工作过程）

1. 模型的输入：

训练时，模型的输入将原始数据随机裁剪成120帧的动作数据，和240帧的音乐数据，监督信号是未来20帧的动作数据。（这里我理解是两帧音乐对应一帧动作，但是在生成未来数据的时候就不需要音乐吗？）。

测试时，输入120帧的舞蹈动作数据作为seed,然后输入音乐，生成了20帧未来动作，却只使用第一帧，并将第一帧append到输入上，循环运行模型。那生成20帧的意义何在？

2. FACT模型：

   FACT的模型结构设计的比较简单，是一个端到端的模型。FACT的整体架构如上图所示，首先，声音特征送入Audio Transformer模块f_{audio}，动作特征送入Motion Transformer模块f_{motion}，这两个模块儿的结构是一样的，都是经典Transformer的encoder部分。结构如图所示

   

   两个模块儿分别生成了一个audio embeddings h^x_{1:T} 和motion embeddings h^y_{1:T'} ,然后将这两部分拼接在一起。然后送入所谓的跨模态Transformer,f_{cross} ,该模块儿直接输出结果，即未来的20帧动作X'。 该模块儿的结构仍然和前两个一样 ，还是经典transformer的full attention encoder模块儿，而不是decoder模块儿 。这里的设计也很疑惑。

   ### 复现效果：

   <见视频>

## [ChoreoMaster : Choreography-Oriented Music-Driven Dance Synthesis](https://netease-gameai.github.io/ChoreoMaster/)

### 1.摘要

该文章是网易AI实验室和清华的合作工作，被SIGGRAPH2021接收的文章，提出了第一个可用于生产的音乐生成舞蹈系统。输入一段音乐，该系统可以生成符合音乐风格，节奏和结构的舞蹈。该论文的主要贡献是设计了一个两阶段的舞蹈生成系统。第一阶段构建了一个基于编舞学的embedding网络，其生成的embedding可以反映舞蹈和音乐之间风格或者节奏的关系。（风格和节奏比较相近的舞蹈和音乐，其生成的embedding之间的欧式距离比较小）。第二阶段构建了一个基于动作图的舞蹈生成系统（图的每个节点是一段舞蹈动作），系统会根据音乐在动作图中找到一条最合适的路径，从而组合成一段舞蹈。

### 2.方法

#### CHOREOGRAPHIC-ORIENTED CHOREOMUSICAL EMBEDDING

风格和节奏对于舞蹈和音乐都有很强的相关性。一种风格和节奏的音乐能够产生对应风格和节奏的舞蹈。所以找到音乐和舞蹈风格和节奏的相关性非常重要。所以本文提出了基于编舞学的choreomusical embedding framework.如下图所示：

![image-20211020183214951](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211020183214951.png)

保持音乐和舞蹈风格的一致性非常重要。而人工对音乐和舞蹈的风格的分类是不准确的，所以本文的核心思想是构建一个choreomusical embdding 网络去对音乐和舞蹈风格之间的联系进行建模。即将音乐和舞蹈映射到同一个特征空间中，相似风格的音乐和舞蹈距离会比较近。

  如上图所示，首先使用没有配对的舞蹈和音乐分别训练两个分类器。其中音乐分类器由4个卷积层和2个GRU层组成；舞蹈动作分类器由4个图卷积和2个GRU层组成。这两个网络的主要作用是将音乐和舞蹈片段压缩映射到一个32维的embedding vector中。

在实际实现中，音乐数据被降采样到16kHZ,使用log-amplitude mel spectograms表示。（computed with 96 mel bins and 160 hop size),舞动动作数据使用18个关节点的位置表示。音乐和舞蹈最短能表达风格的一个单元是一个分乐节phrase.所以作者将舞蹈和音乐片段的长度设置为8s.（在30fps的舞蹈中，截取240帧）.综上，输入数据的shape是$E_M$ [1,96,800].$E_D$ [3,18,240]。

这两个网路在训练过程中分为两阶段训练：

第一阶段：使用所有数据，无需配对分别训练两个网络。

第二阶段：联合训练两个网络，使它们产生的embedding $Z_M$ 和 $Z_D$ 的$L_2$ 距离最小。

这样我们就可以将音乐和舞蹈片段映射到同一个空间中，同一个风格的舞蹈和音乐产生的embedding的欧式距离比较小。

###  Choreomusical Rhythm Embedding

相关乐理知识：

1. rhythm: 旋律

2. meter: 节拍，节拍提供一种框架，用以组织起一系列拍点和节奏，形成更大规模的，由重音和非重音构成的模式。

3. beat：音乐的最小时长单位。

4. bar:通常，meter使用time signature来表示，（比如：2/4, 3/4, 4/4等），分子表示一个bar中有多少个beat,分母表示一个beat的时间长度。

   这个工作需要专业的乐理知识作为支撑，大致意思是，作者请了专业人士，对每段舞蹈中的beat做了人工标记

   ![image-20211021154007338](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211021154007338.png)

如上图所示，一小节的舞蹈有4个beat,使用8个2进制数来表示，其中奇数位置表示半拍。1表示存在beat,0表示不存在beat.按这样算的的话应该有$2^8=256$ 种旋律标记，但是统计数据库发现，旋律标记的类型只有13种。而且还发现，不同舞种之间的旋律标记差别很大。

所以作者设计了一个旋律分类网络，如图三右边所示，由三个模块组成，音乐特征提取网络$E_{MR}$ ,舞蹈特征提取网络$E_{DR}$ , 然后这两个提取出的特征，再拼接上上一个阶段生成的style embedding.送入一个共享的 rhythm signature分类网络，该网络由3个全连接层组成。需要注意的是，这两个特征提取网络的输入，不是原始的音乐和舞蹈底层特征，而是高层特征。音乐网络的输入是 spectral onset strength curve 和 RMS energy curve.

舞蹈网络的输入是 motion kinematic curve, two hand trajectory curvature curves, two foot contact curves.

### CHOREOGRAPHY-ORIENTED DANCE SYNTHESIS

基于编舞学的舞蹈生成。

#### Motion Graph Construction

motion graph是一个有向图，他的每个节点表示数据库中的有单舞蹈动作，每条边表示两个节点之间的转换代价。传统的做法是，每个节点只有一个beat的动作，这样做有两个缺点，（1）meter内部的动作关系被忽略了（2）没有考虑动作间的风格兼容性。为了解决这两个问题，所以本工作每个节点代表一个meter的动作。上一个阶段学到的style embedding和 rythm signature也都被包含在节点中。此外作者还做了一些数据增强的工作，（暂略）。

对于有向图的边，除了考虑传统的转换算是，比如18个关键点的位置，旋转，速度之间的距离。还考虑了两个节点style embedding之间的欧式距离。如果两个节点之间的cost低于某个阈值，我们就会给两个节点添加一条边。

#### Graph-based Optimization

对于舞蹈生成来说，一段生成的舞蹈就对应motion graph中的一条路径。所以我们要做的就是给定一段音乐，找到一条最佳路径和他对应。

![image-20211021181808966](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211021181808966.png)

首先，使用bar detection算法[Gainza 2009].将音乐分成多个bar. 再使用phrases 分割算法将音乐片段分割成不同的phrases[Serra et al ]. 同一个phrase中相似的bar也给分割出来，并赋予一个ID. 综上，一段音乐被分割成多个 bar, 每一个bar给予一个结构标签（ 如上图所示$A_1^1, A_2^1$）。其中A是phrase ID, 上标表示meter ID， 下标表示index.

对于音乐片段$M=\{M_i|i=1,...,n\}$ 中的每一个$M_i$ 求出他的style embedding $Z_{M_i}$ 和 top k rhythm signatures{$R_{M_i}^1...,R_{M_i}^K$}.

我们的目标就是最小化下面这个损失函数：

$C=\lambda_{10} \sum_{i=1}^{n} C_{d}(i)+\lambda_{11} \sum_{i=1}^{n-1} C_{t}(i, i+1)+\zeta \sum_{i<j}^{n} C_{s}(i, j)$ 

$C_d$ 表示data term.表示音乐和舞蹈之间的style embedding和rhythm signature的损失。

$C_t$ 表示节点之间的转移损失。

$C_s$ 表示结构损失，这个损失基于编舞学，就是一般重复的音乐会使用重复的舞蹈动作，  但是在同一个phrase中，相同的meter一般会使用对称的舞蹈动作。所以要加入两个限制（1）同一个phrase,相同meter动作要是对称的（2）不同phrase,相同meter动作要相同。

所以$C_{s}(i, j)=\left\{\begin{array}{lc}
0, & \text { if } D_{i} \text { and } D_{j} \text { satisfy the constraints; } \\
1, & \text { otherwise }
\end{array}\right.$ 

整个过程需要使用一个动态规划算法来求出最优路径[Forney 1973]

### DanceNet3D: Music Based Dance Generation with Parametric Motion Transformer

#### 摘要：

不同于之前的工作，该工作基于动画行业的实践经验，通过预测关键帧之间的动作曲线来生成舞蹈。模型分为两个阶段，第一个阶段根据输入音乐的beats生成关键帧。第二个阶段生成关键帧之间的动作曲线。两阶段使用的都是transformer结构，包含encoder和decoder两部分。但是decoder部分相比经典transfomer做了修改，作者称为MoTrans,该模块中作者引入了Kinematic Chain Networks和Learned Local Attention module.同时，该工作也推出了一个高质量的3d舞蹈数据集，说它是高质量，是因为每一段3d舞蹈都是动画师根据视频制作的（花了18个月），而不是通过动捕或者其他方式生成的，但该数据集只开源了1/3.

#### 方法：

#### 数据集的准备和处理

1.PhantomDance数据集

该数据集搜集了 Niconico、YouTube 上的 300 个热门舞蹈，包含宅舞、嘻哈、爵士、Old School 等多种风格，由专业动画团队在职业舞者指导下，历时 18 个月完成。对比目前学术界来自运动捕捉或真人舞蹈视频的三维重建算法，PhantomDance 在音乐匹配程度、动作优雅程度、艺术表现力上都具有绝对优势。目前该团队公开了其中的 100 个舞蹈 - 音乐数据对，这些数据对组成了 PhantomDance100。

2.关键帧的提取

使用动态规划算法[4]取找出音乐中beat的时间点，然后在抽取出数据集中对应时间的动作作为关键帧，关键帧中使用24个关键点的坐标和旋转表示，（3+4）x 24 = 168维的数据。

3. Motion Curve Formulation

   有了关键帧之后,我们的目标是通过关键帧之间的数据生成动作曲线。因为一个关键点由位置（x,y,z）和旋转（x,y,z,w）组成，每一个坐标需要一个曲线，所以一共需要24x7个曲线。作者使用knots cubic Hermite spline[23]来拟合曲线。（过程略）

关键帧生成：

在测试阶段，我们是没有关键帧数据的，所以需要训练网络来生成关键帧数据。

## 对于每一个beat,我们使用一个以它为中心的0.5s的hamming window.计算声音的MFCC特征并加上12维的chroma 特征。输入到encoder中，

 

参考：

[一文搞懂RNN](https://zhuanlan.zhihu.com/p/30844905)

[深度学习中的注意力机制2017版](https://blog.csdn.net/malefactor/article/details/78767781#commentBox)

[ChoreoMaster 机器之心](https://baijiahao.baidu.com/s?id=1700728830321188017&wfr=spider&for=pc)

[AI Choreographer](https://zhuanlan.zhihu.com/p/404318080)

[乐理知识](https://www.soundbrenner.com/blog/rhythm-basics-beat-measure-meter-time-signature-tempo/)

训练的时候输入120帧舞蹈动作，240帧音乐，输出的监督信号是随后的20帧舞蹈动作。

却在测试的时候希望只输入120帧的示例舞蹈动作，就可以无限的生成剩余的舞蹈动作，而且只拿20帧中的第一帧作为循环输入，为啥不把生成的20帧循环输入？论文说效果不好，为啥会不好呢？



