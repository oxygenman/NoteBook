---
title: 模型压缩之蒸馏论文总结
date: 2019-07-15 18:39:43
tags: 我看的论文
mathjax: true
---

![屏幕快照 2019-07-26 下午3.57.46](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5darvpwhdj31c00u0n5l.jpg)

![image-20190722152711483](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g58nd6kaiqj31c00u0qf8.jpg)

![屏幕快照 2019-07-26 下午3.57.56](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5dasgsm7ij31c00u0qe3.jpg)

![image-20190722152735168](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g58ndmdsuqj31c00u0120.jpg)

![image-20190722152803527](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g58ne2telcj31c00u0agp.jpg)

![屏幕快照 2019-07-26 下午3.58.04](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5dasqnhz9j31c00u07bb.jpg)

## 图像分类

#### 1.2015-NIPS-[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

- 论文摘要：该论文是Hinton对蒸馏概念的诠释，但是第一个提出蒸馏的方法的不是他，而是2104年的[另一篇论文](https://arxiv.org/abs/1312.6184) 。主要思想是将多个模型的支持整合到一个小的模型中，达到模型压缩的目的。该文章还提出了一种新的模型组合方式，使用一个或多个全模型和多个专家模型进行组合，可以达到并行快速训练的效果。

- 实验数据集：MNIST, acoustic model

- 论文思想：
  
  很多昆虫都都有幼虫阶段，从而更易于从生长的环境中吸收能力和维生素，然后变成成虫，更易于日后的种群迁移和繁衍。人也是如此，积累和很多段的人生经验才成就了现在的你。在机器学习算法中，我门大部分也经历了差不多的过程：在训练阶段我们需要从数量大，且存在高度冗余的数据集中提取特征，但是这样的模型病不便于应用，应用的时候需要进化的，提纯的模型。所以说蒸馏就是把重要的东西留下来。使用蒸馏的方法将大模型中的重要知识迁移到小模型中。
  
  那么怎么蒸馏？蒸馏要留下的是大网络学习到的东西，大模型学到的是概率分布，老师要把学到的东西交给学生，使用大网络的输出信息，我们要尽可能的去学习大模型学到的东西。
  
  怎么更好的蒸馏？Hinton认为正常的模型学习到的就是在正确类别上的最大概率，但是不正确的分类上也会得到一些概率。尽管有时候这些概率很小，但是对这些非正确类别的概率也包含了模型泛化的方向信息，包含了特征分布信息。使用这些信息更利于模型的泛化。要想使小模型获得泛化的能力，就要用大模型产生的类别概率，作为小模型的“soft targets”去训练小模型。比如对于MNIST数据集的训练来说，正确类别的概率往往比错误类比的概率大的多，比如在某次识别中，数字7识别为2的概率为$10^{-6}$ ,识别为3的概率为$10^{-9}$ ,这个信息是很有用它表明了2和3很像，但是它对交叉熵损失函数的影响确非常小。 不同与第一篇论文中使用最后一层logits进行训练，Hinton使用softmax输出的概率进行训练，并加入了一个温度参数，使“soft target”更加soft。
  
  如何训练小模型？在训练阶段，可以使用大模型的训练集，也可以使用独立的“transfer set”.大模型可以是单个大模型，也可以是集成模型的概率平均。“soft targets”往往信息熵比较高，所以用它来训练小网络，它需要较少的数据和训练时间，可以使用一个比较高的学习率，算法示意图如下。

[![img](https://camo.githubusercontent.com/cfee36c2ed92f4515c856267d1815cb1632312a7/68747470733a2f2f75706c6f61642d696d616765732e6a69616e7368752e696f2f75706c6f61645f696d616765732f353532393939372d336566303536356132313565333966382e706e673f696d6167654d6f6772322f6175746f2d6f7269656e742f)](https://camo.githubusercontent.com/cfee36c2ed92f4515c856267d1815cb1632312a7/68747470733a2f2f75706c6f61642d696d616765732e6a69616e7368752e696f2f75706c6f61645f696d616765732f353532393939372d336566303536356132313565333966382e706e673f696d6167654d6f6772322f6175746f2d6f7269656e742f)

后续待补。。。。。。。。。。。。。。。。

### 2. 2015-ICLR-[FitNets:Hints for Thin Deep Nets](https://arxiv.org/pdf/1412.6550.pdf)

- **论文摘要：**
  
  论文认为，deep是DNN的主要的功效的来源，于是这篇文章的主要目的是去mimic一个更深但是更小的网络而且准确率比原始网络还要好。那么既然网络很深直接训练会困难，那就通过在中间层加入loss的方法，将网络分成两块来训练。中间的loss则通过teacher的feature map得到。

- **数据集:** CIFAR-10,CIFAR-100,SVHN,AFLW

- **基础网络：** Maxout networks

- **方法：** 
  
  两阶段法：先用hint training 去pretrain小模型前半部分参数，再用KD Training去训练全体参数。
  
  1. Teacher网络的某一中间层的权值为$W_t = W_{hint}$ ,Student 网络的某一中间层的权值为$W_s = W_{guided}$ .使用一个映射函数$W_r$ 来使得$W_{guided}$ 的维度匹配$W_{hint}$ ,得到$W_{s'}$ 。其中对于$W_r$的训练使用MSEloss.
    
     $$
     \mathcal{L}_{H T}\left(\mathbf{W}_{\mathrm{Guided}}, \mathbf{W}_{\mathbf{r}}\right)=\frac{1}{2}\left\|u_{h}\left(\mathbf{x} ; \mathbf{W}_{\mathrm{Hint}}\right)-r\left(v_{g}\left(\mathbf{x} ; \mathbf{W}_{\mathrm{G} \text { uided }}\right) ; \mathbf{W}_{\mathbf{r}}\right)\right\|^{2}
     $$
  
  2. 使用hinton的KD方法训练整个网络
    
     $$
     \mathrm{P}_{\mathrm{T}}^{\tau}=\operatorname{softmax}\left(\frac{\mathrm{a}_{T}}{\tau}\right), \quad \mathrm{P}_{\mathrm{S}}^{\tau}=\operatorname{softmax}\left(\frac{\mathrm{a}_{S}}{\tau}\right)
     $$
     
     $$
     \mathcal{L}_{K D}\left(\mathbf{W}_{\mathbf{S}}\right)=\mathcal{H}\left(\mathbf{y}_{\text { true }}, \mathbf{P}_{\mathrm{S}}\right)+\lambda \mathcal{H}\left(\mathrm{P}_{\mathrm{T}}^{\tau}, \mathrm{P}_{\mathrm{S}}^{\tau}\right)
     $$

- **实验结果：** 
  
  这里只放一个cifar-10的实验作为说明。
  
  ![image-20190721152341678](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57hn81kbwj31am0ig79m.jpg)

- 复现性：开源[Theano](https://github.com/adri-romsor/FitNets)

### 3. 2017-ICLR-[Paying More Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928)

- **论文摘要：**
  
  模仿教师网络的attention map

- **数据集：** CIFAR，Imagenet

- **基础网络结构：** Network in Network, ResNet

- **方法：** 

- 通过网络中间层的attention map，完成teacher network与student network之间的知识迁移。考虑给定的tensor A，基于activation的attention map可以定义为如下三种之一：
  
  ![image-20190721163509466](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57jpyppefj318g070aam.jpg)
  
  随着网络层次的加深，关键区域的attention-level也随之提高。文章最后采用了第二种形式的attention map，取p=2，并且activation-based attention map的知识迁移效果优于gradient-based attention map，loss定义及迁移过程如下：
  
  $$
  \mathcal{L}_{A T}=\mathcal{L}\left(\mathbf{W}_{S}, x\right)+\frac{\beta}{2} \sum_{j \in \mathcal{I}}\left\|\frac{Q_{S}^{j}}{\left\|Q_{S}^{j}\right\|_{2}}-\frac{Q_{T}^{j}}{\left\|Q_{T}^{j}\right\|_{2}}\right\|_{p}
  $$
  
  $$
  Q_{S}^{j}=\operatorname{vec}\left(F\left(A_{S}^{j}\right)\right) \text { and } Q_{T}^{j}=\operatorname{vec}\left(F\left(A_{T}^{j}\right)\right)
  $$
  
  ![image-20190721164008950](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57jurpzcqj31bq0h0jtv.jpg)

- **实验结果：** 
  
  cifar-10
  
  ![image-20190721164521204](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57k0b2rvtj31ba0bggms.jpg)

- **复现性**：[开源](https://github.com/szagoruyko/attention-transfer) 

### 4. 2017-CVPR-[A Gift from Knowledge Distillation:Fast Optimization, Network Minimization and Transfer Learning](https://zpascal.net/cvpr2017/Yim_A_Gift_From_CVPR_2017_paper.pdf)

- **论文摘要：** 
  
  定义需要从教师网络学习的知识，为flow between layers, 通过两层之间的内积来计算。

- **数据集：** CIFAR-10, CIFAR-100

- **基础网路：** res-net

- **方法：**
  
  需要学习的知识可表示为训练的求解过程（FSP: Flow of the Solution Procedure），教师网络或学生网络的FSP矩阵定义如下（Gram形式的矩阵）：
  
  $$
  G_{i, j}(x ; W)=\sum_{s=1}^{h} \sum_{t=1}^{w} \frac{F_{s, t, i}^{1}(x ; W) \times F_{s, t, j}^{2}(x ; W)}{h \times w}
  $$
  
  ![image-20190721171333348](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57ktjamrvj30sy0g2mzr.jpg)p

训练第一阶段：最小化教师网络FSP矩阵和学生网络FSP矩阵之间的L2 Loss.

$$\begin{array}{l}{L_{F S P}\left(W_{t}, W_{s}\right)}  {=\frac{1}{N} \sum_{x} \sum_{i=1}^{n} \lambda_{i} \times\left\|\left(G_{i}^{T}\left(x ; W_{t}\right)-G_{i}^{S}\left(x ; W_{s}\right) \|_{2}^{2}\right.\right.}\end{array}$$

训练第二阶段：在目标任务的数据集上fine-tune学生网络。从而达到知识迁移，快速收敛，以及迁移学习的目的。

![image-20190721173928639](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57lki10knj31gs0u0tgk.jpg)

- **实验结果：**
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57lwhafgbj30tk0cs763.jpg" style="zoom:50%">
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57ly168sdj30t60ck0uj.jpg" style="zoom:50%">

- 复现性：h

### 5.[2017-arXiv-Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/pdf/1707.01219.pdf)(图森)

- **论文摘要**
  
  该论文主要引入了最大平均差异 Maximum Mean Discrepancy(MMD),来计算教师网络和学生网络的特征分布差异，并与原始损失函数结合，提高学生网络的性能。(定义新的知识，神经元选择性NST，或者激活值模式，反映了神经元的特征选择。)

- **数据集**
  
  CIFAR-10,CIFAR-100, ImageNet LSVRC2012.

- **方法**
  
  Neuron Selectivity Transfer (NST)
  
  为什么不直接匹配feature maps?因为它忽视了空间中样本的密度。
  
  NST Loss的定义为：
  
  $$
  \mathcal{L}_{\mathrm{NST}}\left(\mathbf{W}_{S}\right)=\mathcal{H}\left(\boldsymbol{y}_{\text { true }}, \boldsymbol{p}_{S}\right)+\frac{\lambda}{2} \mathcal{L}_{\mathrm{MMD}^{2}}\left(\mathbf{F}_{T}, \mathbf{F}_{S}\right)
  $$
  
  $\mathcal{H}$ 表示标准交叉熵损失函数。

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g51k7g3xa9j313o0l60tr.jpg" style="zoom:50%">       多项式核：$k(\boldsymbol{x}, \boldsymbol{y})=\left(\boldsymbol{x}^{\top} \boldsymbol{y}+c\right)^{d}$ 多项式核反映了区域相似性。

​      高斯核：$k(\boldsymbol{x}, \boldsymbol{y})=\exp \left(-\frac{\|\boldsymbol{x}-\boldsymbol{y}\|_{2}^{2}}{2 \sigma^{2}}\right)$

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g53ren7gzuj31cm0lgdh1.jpg" style="zoom:50%">

- **实验结果**
  
  <img src = "https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g51mqkr1wzj30yo0kiaek.jpg" style="zoom: 50%">
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g51mrmkldaj30nc0fkwhj.jpg" style="zoom:50%">
  
  ![屏幕快照 2019-07-16 下午1.52.01](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g549nghgotj31am0bygoy.jpg)

### 6. 2017-DarkRank:Acclerating Deep Metric Learning Via Cross Sample Similarities Transfer（图森）

- **论文摘要：**
  
  改论文定义的需要从教师网络学习到的知识为样本间的相似性排序融入到监督训练中，并融合了softmax,Verify loss, triplet loss共同训练。

- **数据集：** CUHK03, Market1501,CUB-200-2011

- **网络模型：** Inception-BN, NIN-BN
  
  方法：
  
  ![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57s3augprj30y10jkjtz.jpg)

- 实验结果：
  
  <img src= "https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57sj9fiaej30tg0nogpq.jpg" style="zoom: 50%">
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57slsxg8wj30ty0hwtba.jpg" style="zoom:50%">

- 复现性：开源[MxNet](https://github.com/TuSimple/DarkRank)

### 7.  [2019-CVPR-Relational Knowledge Distillation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf)

- **论文摘要：** 
  
  本文定义了一种新的需要从教师网络学习的知识，即样本之间的关系，（relational knowledge distillation, RKD).提出了两种衡量样本关系的方式，distance-wise, angel-wise.

- **数据集：** CIFAR-100, Tiny ImageNet.

- **基础网络：** Resnet50, VGG11

- **方法：** 
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57mvel4z0j30q40m0wkr.jpg" style="zoom:50%"> 
  
  定义新的损失函数：
  
  $$
  \mathcal{L}_{\mathrm{RKD}}=\sum_{\left(x_{1}, \ldots, x_{n}\right) \in \mathcal{X}^{N}} l\left(\psi\left(t_{1}, . ., t_{n}\right), \psi\left(s_{1}, \dots, s_{n}\right)\right)
  $$
  
  其中$（x_1,x_2,...x_n）$ 是 n个样本组成的关系元组。$\psi$ 是relational potential function.
  
  - Distance wise distillation loss
    
    $$
    \psi_{\mathrm{D}}\left(t_{i}, t_{j}\right)=\frac{1}{\mu}\left\|t_{i}-t_{j}\right\|_{2}
    $$
  
  $$
  \mathcal{L}_{\mathrm{RKD}-\mathrm{D}}=\sum_{\left(x_{i}, x_{j}\right) \in \mathcal{X}^{2}} l_{\delta}\left(\psi_{\mathrm{D}}\left(t_{i}, t_{j}\right), \psi_{\mathrm{D}}\left(s_{i}, s_{j}\right)\right)
  $$
  
    Huber Loss
  
  $$
  l_{\delta}(x, y)=\left\{\begin{array}{ll}{\frac{1}{2}(x-y)^{2}} & {\text { for }|x-y| \leq 1} \\ {|x-y|-\frac{1}{2},} & {\text { otherwise }}\end{array}\right.
  $$

- Angle-wise distillation loss
  
  $$
  \begin{array}{l}{\psi_{\mathrm{A}}\left(t_{i}, t_{j}, t_{k}\right)=\cos \angle t_{i} t_{j} t_{k}=\left\langle\mathbf{e}^{i j}, \mathbf{e}^{k j}\right\rangle} \\ {\text { where } \quad \mathbf{e}^{i j}=\frac{t_{i}-t_{j}}{\left\|t_{i}-t_{j}\right\|_{2}}, \mathbf{e}^{k j}=\frac{t_{k}-t_{j}}{\left\|t_{k}-t_{j}\right\|_{2}}}\end{array}
  $$
  
  $$
  \mathcal{L}_{\mathrm{RKD}-\mathrm{A}}=\sum_{\left(x_{i}, x_{j}, x_{k}\right) \in \mathcal{X}^{3}} l_{\delta}\left(\psi_{\mathrm{A}}\left(t_{i}, t_{j}, t_{k}\right), \psi_{\mathrm{A}}\left(s_{i}, s_{j}, s_{k}\right)\right)
  $$

- 实验结果
  
  teacher:resnet-50, student:VGG11
  
  teacher:resnet-101, resnet-18
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g57qrao0jqj30u40i4t9x.jpg"  style="zoom: 50%"> 

- 复现难度：h

## 目标检测

### 1.[2017-NIPS-Learning Efficient Object Detection Models with Knowledge Distillation](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf)（[博客解读](https://blog.csdn.net/nature553863/article/details/82463249)）

- **论文摘要**
  
  本文主要工作是将Knowledge Distillation和Hint Learning技术用在了目标检测模型Faster-RCNN上，提出了weighted cross-entropy来解决类别不平衡问题，使用teacher bounded loss解决bounding box 的回归问题，使用hint learning来学习教师网络的层间特征。

- **数据集**
  
  PASCAL, KITTI, ILSVRC, MS-COCO

- **方法**
  
  学生网络的整体目标函数：
  
  $$
  \begin{aligned} L_{R C N} &=\frac{1}{N} \sum_{i} L_{c l s}^{R C N}+\lambda \frac{1}{N} \sum_{j} L_{r e g}^{R C N} 
  \\\\ L_{R P N} &=\frac{1}{M} \sum_{i} L_{c l s}^{R P N}+\lambda \frac{1}{M} \sum_{j} L_{r e g}^{R P N} 
  \\\\ L &=L_{R P N}+L_{R C N}+\gamma L_{H i n t} \end{aligned}
  $$
  
  $L_{R C N},L_{RPN},L_{Hint}$分别代表RCN网络，RPN网络和Hint-based损失函数。
  
  1. 对于分类的知识蒸馏（with Imabalanced Classes）
    
     $$
     L_{c l s}=\mu L_{h a r d}\left(P_{s}, y\right)+(1-\mu) L_{s o f t}\left(P_{s}, P_{t}\right)
     $$
     
     $$
     L_{s o f t}\left(P_{s}, P_{t}\right)=-\sum w_{c} P_{t} \log P_{s}
     $$
     
     其中$P_t=softmax(\frac{Z_t}{T})$ 表示教师网络的输出，T是一个温度参数，这里通常设为1.$P_{s}=\operatorname{softmax}\left(\frac{Z_{s}}{T}\right)$ 表示学生网络的输出。$L_{cls}$为groundtrues类别标签.这里的损失函数都采用的交叉熵，为了解决目标检测任务中类别不平衡的问题，引入了加权交叉熵，对背景类采用较大的权重，因为背景类在实验过程中，分错的概率较小。
  
  2. 对于回归知识的蒸馏（with Teacher Bounds）
    
     $$
     L_{b}\left(R_{s}, R_{t}, y\right) =\left\{\begin{array}{ll}{\left\|R_{s}-y\right\|_{2}^{2},}   {\text { if }\left\|R_{s}-y\right\|_{2}^{2}+m>\left\|R_{t}-y\right\|_{2}^{2}} \\ {0,}  {\text { otherwise }}\end{array}\right.
     $$
     
     $$
     L_{\text {reg}} =L_{s L 1}\left(R_{s}, y_{\text {reg}}\right)+\nu L_{b}\left(R_{s}, R_{t}, y_{\text {reg}}\right)
     $$
     
     对于boundingbox的回归不同于对离散类别信息的蒸馏，它本来就是连续的。而教师网络提供的回归方向有可能和groundtruth的方向是相反的，所以这里将教师的信息，作为学生网络回归的上边界，当超过一定距离时，才对学生网络提供监督。
  
  3. Hint Learning with Feature Adaptation
    
     即让学生网络去学习教师网络中间层特征图的分布，可以通过计算$L_1和L_2$loss实现。
     
     $$
     L_{H i n t}(V, Z)=\|V-Z\|_{2}^{2}
     $$
     
     $$
     L_{H i n t}(V, Z)=\|V-Z\|_{1}
     $$

- **实验结果**
  
  ![屏幕快照 2019-07-16 上午10.54.04](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g51hrfjvvaj312y0eoq7c.jpg)
  
  下图是使用高分辨率图片训练教师网络，低分辨率图片训练学生网络的结果。
  
  ![屏幕快照 2019-07-16 上午11.05.15](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g51i305natj31cu0cqjv1.jpg)

### 2. [2017-商汤-Mimicking Very Efficient Network for Object Detection](http://xueshu.baidu.com/usercenter/paper/show?paperid=29f43e4a7e341d555f0f95566bde9cd4&site=xueshu_se)

- 论文摘要：
  
  论文针对目标检测模型，论文提出了一种针对特征图Mimicking(蒸馏)的方法，主要针对localregion进行蒸馏，而不是全局特征。小型化的Inception取得了2.5倍的压缩，并取得和原始模型相当的准确度。在Caltech数据集上处理1000x1500的输入可以达到80FPS。

- 数据集：Caltech, Pascal VOC

- 基础模型：Inception, ResNet，Faser-rcnn, R-FCN.

- 方法：
  
  1. 目标检测产生的特征图往往维度比较高，难以直接对两个feature map进行回归。而且对于目标检测特征图中，只有object区域的相应比较高，背景区域都是噪声。所以在特征图中，只有目标附近区域（local regions）才包含较多的有用信息。只针对RPN网络。
    
     $$
     \begin{aligned} \mathcal{L}(W) &=\lambda_{1} \mathcal{L}_{m}(W)+\mathcal{L}_{g t}(W) \\ \mathcal{L}_{m}(W) &=\frac{1}{2 N} \sum_{i}\left\|u^{(i)}-r\left(v^{(i)}\right)\right\|_{2}^{2} \\ \mathcal{L}_{g t}(W) &=\mathcal{L}_{c l s}(W)+\lambda_{2} \mathcal{L}_{r e g}(W) \end{aligned}
     $$
     
     $\mathcal{L}(W)$ 表示特征图模仿的$L_2$ 损失，N表示region proposal的数量，$u^{i}$表示教师网络产生的第i个proposal经过spp处理后得到的特征，$v{i}$同理，r是一个回归函数用来统一两者的维度。
     
     这样还存在问题，一是特征图的的值可能较大，需要小心平衡，groundtruth损失和mimic损失，即$\lambda$ 参数；二是spp可能会破坏特征。所以进行了归一化改进，和去除spp,不同rp对应的特征图的大小是不同的。.
     
     $$
     \mathcal{L}_{m}(W)=\frac{1}{2 N} \sum_{i} \frac{1}{m_{i}}\left\|u^{(i)}-r\left(v^{(i)}\right)\right\|_{2}^{2}
     $$
2. Two-stage Mimic: 为了提高准确率除了对RPN网络进行Mimic,对detecter网络也进行Mimic.

3. Mimic over Scales:使用大图片去训练教师网络，使用小图片训练学生网络
- 实验结果
  
  Caltech,评价指标：log average missrate on FPPI
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g53yo475yaj30sc0ccq51.jpg" style="zoom:50%">

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g53yqax3mlj30u20n2tci.jpg" style="zoom:50%">

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g53ytkx7q6j30te0isgp2.jpg" style="zoom:50%">

减小输入分辨率：

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5c68zwqb9j30io08k3zt.jpg" style="zoom:50">

Pascal VOC 评价指标：mAP

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g53yzm5xpwj30u00c0tam.jpg" style="zoom:50%">

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g53yzxrpbej30sm0gk76z.jpg" style="zoom:50%">

- 复现难度：m

### 3. [2018-商汤-CVPR-2019-Quantization Mimic: Towards Very Tiny CNN for Object Detection](https://arxiv.org/pdf/1805.02152.pdf)

- 论文摘要：本文主要目标是训练VeryTiny网络，（VGG，1/32）.提出了Quantization Mimic方法来蒸馏网络，首先量化教师网络，再将学生网络使用蒸馏方法Mimic教师网络。量化教师网络可以降低学生网络的参数搜索空间，别人使用量化直接压缩网络，作者使用量化来辅助蒸馏。

- 数据集：WIDER FACE， PascalVOC

- 基础网络：VGG with R-FCN，ResNet with Faster-R-CNN.

- 方法：
  
  ![image-20190718151724599](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g540ls258aj31820tsgrz.jpg)
  
  1. 量化：对教师网络的最后一层特征进行量化，使用Uniform量化，因为他更适用于当前模型。
    
     量化函数$Q$为：
     
     $$
     Q(f)=\beta \quad \text { if } \frac{\alpha+\beta}{2}<f \leq \frac{\gamma+\beta}{2}
     $$
     
     其中，$\alpha,\beta, \gamma$ 是在量化字典中相邻的元素：
     
     $$
     D=\{0, s, 2 s, 3 s . .\}
     $$
     
     s是量化步长。
     
     需要说明的是，在训练的时候类似于BNN，使用的是全精度梯度。
  
  2. Mimic:采用和上一篇论文一样的方法
    
     $$
     \begin{array}{c}{L=L_{c l s}^{r}+L_{r e g}^{r}+L_{c l s}^{d}+L_{r e g}^{d}+\lambda L_{m}} \\ {L_{m}=\frac{1}{2 N} \sum_{i}\left\|f_{t}^{i}-r\left(f_{s}^{i}\right)\right\|_{2}^{2}}\end{array}
     $$
3. Quantization Mimic:
     $
     L_{m}=\frac{1}{2 N} \sum_{i}\left\|Q\left(f_{t}^{i}\right)-Q\left(r\left(f_{s}^{i}\right)\right)\right\|_{2}^{2}
     $
     ![image-20190718153806956](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5417b20v3j31180scn42.jpg)
- [CSDN解读](https://blog.csdn.net/bryant_meng/article/details/83056203)

- 实验结果：
  
  1. WIDER FACE Dataset
    
     ![image-20190718154943309](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g541jdkwhrj30v70u07bl-20220717190330356.jpg)

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g541lu1m7dj310c0e076z.jpg" style="zoom:50%">

2. Pascal VOC
  
   ![image-20190718155419344](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g541o5xe3zj31040cu76i.jpg)
   
   效果有点儿差哦,有一定的参考价值。
- 复现难度：h 

### 4. 2019-华为诺亚方舟-[Distilling Object Detectors with Fine-grained Feature Imitation](https://arxiv.org/pdf/1906.03609.pdf)

- 论文摘要
  
  本文的思想和商汤的第一篇论文很相似，认为在目标检测模型的模型蒸馏中不能单纯的mimic教师网络的整个特征图，这样会引入很多噪声，目标附近的区域才是比较重要的，但是本文不是去mimic region proposal,而是事先根据gt和anchor计算出目标临近的区域。然后让学生去mimic这些区域的特征。

- 数据集：KITTI, Pascal VOC, COCO.

- 模型：toy-detector, faster R-CNN.

- 方法：
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g543cgs196j30to0si44j.jpg" style="zoom:50%">
  
  1. 如图所示，首先计算特征图上每个点所有anchors和gtbox的IOU，然后计算出最大的IOU值，$M = max(m)$ .然后乘一个阈值因子$\psi$ ,然后得到一个阈值$F=\psi * M$ .用这个阈值去过滤IOU map,得到一个WXH的map,最后结合所有的gtbox得到一个mask I.
  
  2. $N_p$是mask上正值得个数，$f_{adap}$ 是为了对齐两个网络的特征图。
  
  3. $$
     \begin{aligned} L_{i m i t a t i o n} &=\frac{1}{2 N_{p}} \sum_{i=1}^{W} \sum_{j=1}^{H} \sum_{c=1}^{C} I_{i j}\left(f_{\mathrm{adap}}(s)_{i j c}-t_{i j c}\right)^{2} \\ \text { where } N_{p} &=\sum_{i=1}^{W} \sum_{j=1}^{H} I_{i j} \end{aligned}
     $$
     
     总的损失函数为：

$$
L=L_{g t}+\lambda L_{i m i t a t i o n}
$$

- 实验结果：
  
  1. toy-detector
    
     <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5449r87amj31mo0q6wh0.jpg">
2. Faster R-CNN
  
   ![image-20190718172955271](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g544fo2zu5j31iq0u0qay.jpg)
- 复现难度：[开源](https://github.com/twangnh/Distilling-Object-Detectors)

## 人脸识别

### 1.[2016-AAAI-汤晓鸥组-Face Model Compression by Distilling Knowledge from Neurons](http://personal.ie.cuhk.edu.hk/~pluo/pdf/aaai16-face-model-compression.pdf)

- **论文摘要：** 
  
  不同于Hinton使用“soft target”作为需要学习的知识，本文使用高层的神经元作为学习的知识，它含有和输出概率同等的信息，但是更加的compact(坚实，紧凑)，而且使用“soft target”的话不容易拟合。利用学习到的人脸的基本特点(领域知识)，提出了一个神经元选择方法选择出和人脸识别更相关的神经元。使用选择出来的神经元作为监督信息来模仿DeepID2+和DeepID3。并在LFW上取得了比教师网络的更高的精度。当使用一个DeepID2+的集成网络时（6个网络），学生网络可以取得51.6倍的压缩比，和90倍的推理速度提升。AUC为98.43。

- **数据集：** training: WDRef, CelebFaces+; testing: LFW

- **基础网络：** DeepID2+, DeepID3

- **方法：** 
  
  1. 神经元选择方法基于三个original observations(domain knowledge):
    
     - 深度学习学习到的人脸特征是人脸属性的分布特征（distributed representation over face attributes),包括身份有关 属性(IA)，比如性别，种族。和身份无关属性(NA)，比如表情，光照，照片质量等。在训练过程中尽管没有提供这些属性信息，但是可以发现某个神经元和某些属性是有联系的。
     - 这些分布式特征既不是不变的，也不是完全分离的（neither invariant nor completely factorized）.应该将与NA有关的神经元移除。
     - 有些神经元一直处于抑制状态，是噪音。
     
     一个平均场算法（mean field algorithm),可以让我们选出与IA有关的神经元，但是相互关系较少的神经元。
  
  2. 使用Neuron Selection来训练学生网络
    
     $$
     L(\mathcal{D})=\frac{1}{2 M} \sum_{i=1}^{M}\left\|\mathbf{f}_{i}-g\left(\mathbf{I}_{i} ; \mathbf{W}\right)\right\|_{2}^{2}
     $$
     
     上式为训练学生网络的损失函数，其中$f_i$为地i个图像通过神经元选择方法筛选出来的特征。
  
  3. 神经元如何选择？
    
     将神经元之间的关系看成一个全连接图问题，比如N个Neuron $\mathbf{y}=\left\{y_{i}\right\}_{i=1}^{N}$,$y_{i} \in\{0,1\}$ .
     
     通过最小化下面的能量函数来实现神经元的选择：
     
     $$
     E(\mathbf{y})=\sum_{i=1}^{N} \Phi\left(y_{i}\right)+\lambda \sum_{i=1}^{N} \sum_{j=1, j \neq i}^{N} \Psi\left(y_{i}, y_{j}\right)
     $$
     
     其中$\Phi(y_{i})$ 和 $\Psi(y_i,y_j)$ 分别 表示选中神经元i的损失，和同时选中i，j的损失。
     
     $\Phi(x_i) = f(x_i)$ ,$f(.)$ 是一个惩罚函数，$x_i$ 是一个向量，用来表示神经元i的特征区分能力。
     
     $\Psi(.)$ 也是一个惩罚函数，用来惩罚相关性较大的神经元。
     
     $$
     \Psi\left(y_{i}, y_{j}\right)=\exp \left\{-\frac{1}{2}\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}^{2}\right\}
     $$
     
     这个能量函数可以用平均场算法求解。
  
  4. $x_i$ 怎么求得？
    
     $x_i$的每一维表示第i个神经元，对第j个特征的分类准确率：
     
     $$
     \text { i.e. } \forall \mathbf{x}_{i} \in \mathbb{R}^{1 \times 40} \text { and } \mathbf{x}_{i(j)}=\frac{T P_{j}+T N_{j}}{2}
     $$
     
     其中，$TP_j和TN_j$ 表示真正率和真假率。至于这个怎么统计的，文中没有提，我觉得因该是根据某个样本的属性，看这个神经元对这个样本有没有反应来统计。
     
     最后，得到$f(.)$的最终表达式：
     
     $$
     f\left(\mathbf{x}_{i}\right)=\frac{\max \left\{\mathbf{x}_{i(j)}\right\} \forall j \in \mathrm{NA}-\operatorname{avg}\left\{\mathbf{x}_{i(j)}\right\} \forall j \in \mathrm{NA}}{\max \left\{\mathbf{x}_{i(j)}\right\} \forall j \in \mathrm{IA}-\operatorname{avg}\left\{\mathbf{x}_{i(j)}\right\} \forall j \in \mathrm{IA}}
     $$
     
     如果一个神经元对NA属性的选择性大于对IA的选择性，他就会受到惩罚。

- 实验结果
  
  在LFW数据集上进行人脸验证，人脸验证通过计算欧式距离来实现。评价指标AUC。
  
  ![image-20190719112633318](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g54zjxb288j31800d2dhw.jpg)

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g54zkhl1dhj30nc0pmq64.jpg" style="zoom:50%">

- 复现难度：[开源](https://github.com/liuziwei7/mobile-id)
  
  ------

### 2. [2019-arXiv-ICML-Triple Distillation for Deep Face Recognition](https://arxiv.org/abs/1905.04457?context=cs.CV)

- **论文摘要：**
  
  本文主要把Triplet loss和蒸馏的思想进行了结合引入了Triplet-distillation.改进了Triplet-loss中，identities之间的固定间距。从训练好教师网络中学习indenties之间的多样性知识。

- **数据集：** LFW，AgeDB, CPLFW

- **网络模型：** ResNet-100, slim version of MobileFaceNet.

- **方法：** 
  
    原始Triplet loss: 
  
  $$ \mathcal{L}=\frac{1}{N} \sum_{i}^{N} \max \left(\mathcal{D}\left(x_{i}^{a}, x_{i}^{p}\right)-\mathcal{D}\left(x_{i}^{a}, x_{i}^{n}\right)+m, 0\right) $$
  
   在原始Triplet loss中，对于所有的identities m是相同的且固定不变的，所有的聚簇都将使用固定的距离粗鲁的分开，它忽视了identities之间微妙的相似性。比如说A和B的相似性大于A和C的相似性，那么理论上{A,B}的m应该小于{B,C}的m.和hinton的思想一样的，这样的相似性是有用的。 $$ \begin{array}{c}{\mathcal{L}=\frac{1}{N} \sum_{i}^{N} \max \left(\mathcal{D}\left(x_{i}^{a}, x_{i}^{p}\right)-\mathcal{D}\left(x_{i}^{a}, x_{i}^{n}\right)+\mathcal{F}(d), 0\right)} \\ {d=\max \left(\mathcal{T}\left(x_{i}^{a}, x_{i}^{n}\right)-\mathcal{T}\left(x_{i}^{a}, x_{i}^{p}\right), 0\right)}\end{array} $$ 
  
  Triplet Distillation:
  
  先训练一个教师网络，然后教师网络提取identities的特征，计算它们之间的距离。如上所示$\mathcal{D}$代表学生网络计算的距离，$\mathcal{T}$代表教师网络的距离。
  
  $\mathcal{F}$是一个简单的线性函数: $\mathcal{F}(d)=\frac{m_{\max }-m_{\min }}{d_{\max }} d+m_{\min }$
  
  通过这种方式m被限制在了$m_{min}$ 和 $m_{max}$ 之间。

- **实验结果：** 
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g555d2woo1j30j605474v.jpg" style="zoom:70%">
  
  <img src="http://ww3.sinaimg.cn/large/006tNc79ly1g555gehz77j30iu09m0u1.jpg" style="zoom:50%">

- 复现难度：将要开源

### 3.2019-arXiv-[Deep Face Recognition Model Compression via Knowledge Transfer and Distillation](https://arxiv.org/abs/1906.00619)

- **论文摘要:** 降低输入图像的大小来压缩网络，通过蒸馏方法来提高精度。

- **数据集：** LFW

- **实验结果：** 
  
  <img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/006tNc79ly1g5c44c953uj30j40ayabb.jpg"  style="zoom:50%">

### 开源统计：

[MaskRcnn with Knowledge Distillation (pytorch)](https://github.com/RuiChen96/MaskRCNN-PyTorch) 

[Distillation-of-Faster-rcnn](https://github.com/HqWei/Distillation-of-Faster-rcnn)

[各种蒸馏方法总结实验对比](https://github.com/sseung0703/Knowledge_distillation_methods_wtih_Tensorflow)

[Relational knowledge distillation](https://github.com/lenscloth/RKD)

[Structured Knowledge Distillation for Semantic Segmentation](https://github.com/irfanICMLL/structure_knowledge_distillation)

[Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://github.com/bhheo/AB_distillation)

[Knowledge Distillation with Adversarial Samples Supporting Decision Boundary (AAAI 2019)](https://github.com/bhheo/BSS_distillation)