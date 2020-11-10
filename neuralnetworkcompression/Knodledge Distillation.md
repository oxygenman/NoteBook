## Knodledge Distillation

### 一、知识蒸馏相关论文

### [CSDN博客总结1](https://blog.csdn.net/qzrdypbuqk/article/details/81482598) ：

   从模型压缩的角度掉了有关蒸馏相关的进展。从有无开源或者第三方实现进行了分类，按时间顺序进行了介绍。

### [github开源知识蒸馏总结 awesome-knowldge-distillation](https://github.com/dkozlov/awesome-knowledge-distillation)

实时更新模型蒸馏最新进展，但是没有分类，目前看到的最全的相关总结。

### [CSDN博客总结2](https://blog.csdn.net/nature553863/article/details/80568658)

### 二、论文阅读：

#### 1.2015-NIPS《Distilling the Knowledge in a Neural Network》

- 论文摘要：该论文是Hinton对蒸馏概念的诠释，但是第一个提出蒸馏的方法的不是他，而是2104年的[另一篇论文](http://xueshu.baidu.com/s?wd=paperuri%3A%28078415e6ab570770529798299e0d8b90%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.arxiv.org%2Fpdf%2F1312.6184v1.pdf&ie=utf-8&sc_us=7007594391503052629) 。主要思想是将多个模型的支持整合到一个小的模型中，达到模型压缩的目的。该文章还提出了一种新的模型组合方式，使用一个或多个全模型和多个专家模型进行组合，可以达到并行快速训练的效果。

- 实验数据集：MNIST, acoustic model

- 论文思想：

  很多昆虫都都有幼虫阶段，从而更易于从生长的环境中吸收能力和维生素，然后变成成虫，更易于日后的种群迁移和繁衍。人也是如此，积累和很多段的人生经验才成就了现在的你。在机器学习算法中，我门大部分也经历了差不多的过程：在训练阶段我们需要从数量大，且存在高度冗余的数据集中提取特征，但是这样的模型病不便于应用，应用的时候需要进化的，提纯的模型。所以说蒸馏就是把重要的东西留下来。使用蒸馏的方法将大模型中的重要知识迁移到小模型中。

  那么怎么蒸馏？蒸馏要留下的是大网络学习到的东西，大模型学到的是概率分布，老师要把学到的东西交给学生，使用大网络的输出信息，我们要尽可能的去学习大模型学到的东西。

  怎么更好的蒸馏？Hinton认为正常的模型学习到的就是在正确类别上的最大概率，但是不正确的分类上也会得到一些概率。尽管有时候这些概率很小，但是对这些非正确类别的概率也包含了模型泛化的方向信息，包含了特征分布信息。使用这些信息更利于模型的泛化。要想使小模型获得泛化的能力，就要用大模型产生的类别概率，作为小模型的“soft targets”去训练小模型。比如对于MNIST数据集的训练来说，正确类别的概率往往比错误类比的概率大的多，比如在某次识别中，数字7识别为2的概率为$10^{-6}$ ,识别为3的概率为$10^{-9}$ ,这个信息是很有用它表明了2和3很像，但是它对交叉熵损失函数的影响确非常小。 不同与第一篇论文中使用最后一层logits进行训练，Hinton使用softmax输出的概率进行训练，并加入了一个温度参数，使“soft target”更加soft。

  如何训练小模型？在训练阶段，可以使用大模型的训练集，也可以使用独立的“transfer set”.大模型可以是单个大模型，也可以是集成模型的概率平均。“soft targets”往往信息熵比较高，所以用它来训练小网络，它需要较少的数据和训练时间，可以使用一个比较高的学习率，算法示意图如下。

![img](https://upload-images.jianshu.io/upload_images/5529997-3ef0565a215e39f8.png?imageMogr2/auto-orient/)

后续待补。。。。。。。。。。。。。。。。

#### 2.2016-AAAI-汤晓鸥组《Face Model Compression by Distilling Knowledge from Neurons》

- 论文摘要：不同于Hinton使用“soft target”作为需要学习的知识，本文使用高层的神经元作为学习的知识，它含有和输出概率同等的信息，但是更加的compact(坚实，紧凑)，而且使用“soft target”的话不容易拟合。利用学习到的人脸的基本特点，提出了一个神经元选择方法选择出和人脸识别更相关的神经元。使用选择出来的神经元作为监督信息来模仿DeepID2+和DeepID3。并在LFW上取得了比教师网络的更高的精度。当使用一个DeepID2+的集成网络时，学生网络可以取得51.6倍的压缩比，和90倍的推理速度提升。AUC为98.43。

- 论文思想：本文认为虽然和hard-target相比，softmax输出的概率含有更丰富的信息，但是因为经过softmax层后，很多概率接近0，造成了很多信息的损失。[Ba和Caruana](http://xueshu.baidu.com/s?wd=paperuri%3A%28078415e6ab570770529798299e0d8b90%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.arxiv.org%2Fpdf%2F1312.6184v1.pdf&ie=utf-8&sc_us=7007594391503052629) 使用logits来解决这个问题，但是由于没有对logits进行限制，它可能含有一些过大的值，就引入了一些噪声。而Hinton引入了温度参数来解决这个问题，对噪声进行惩罚。Hinton虽然取得了成功，但是通过实验发现这种做法并不适用于人脸识别。因为人脸识别标签维度较高，很难拟合和收敛？（为什么？？）所以本文选择隐藏层重要的神经元作为监督信息，这样的选择基于三个观察（domain knowledge）: 1.深度学习学习到的人脸特征是人脸属性的分布特征（distributed representation over face attributes),包括身份有关    属性（IA），比如性别，种族。和身份无关属性(NA)，比如表情，光照，照片质量等。在训练过程中尽管没有提供这些属性信息，但是可以发现某个神经元和某些属性是有联系的。

  2.这些分布式特征既不是不变的，也不是完全分离的（neither invariant nor completely factorized）.某些神经元跟IA和NA都有关应该将它们分离开来。

   3.有些神经元一直处于抑制状态，是噪音。

- 论文主要贡献

  1.证明了在人脸识别时，监督信号越compact,网络收敛越高效。因为标签维度太大使用“soft target”训练学生网络难以拟合。所以使用最高隐藏层神经元提取的特征作为监督信号。

  2.揭露了三个与深度网络人脸特征有关的观察。用来识别神经元得到的特征信息是否有效。

  3.根据观察规则，提出了一个神经元选择的方法用来选取有用的神经元，对网络进行监督。

- 采用数据集：LFW

  未完待续。。。。。。。

#### 3.2019-arXiv-《Deep Face Recognition Model Compression via Knowledge Transfer and Distillation》

- 论文摘要：本文做蒸馏的主要思想是使用分辨率较高的图片训练大网络，然后在用分辨率较低的图片和大网络的最后一层的pooling层的信息来监督小网络的训练。本文还使用了知识迁移的方法（knowledge transfer）与知识蒸馏相结合来提高网络的精度，所以本文的学生网络并没有改变教师网络的网络结构。
- 采用数据集：LFW IJB-C

#### 4.2019-arXiv-ICML-《Triplet Distillation for Deep Face Recognition》

- 论文摘要：本文主要把Triplet loss和蒸馏的思想进行了结合引入了Triplet-distillation.改进了Triplet-loss中，identities之间的固定间距。从训练好教师网络中学习indenties之间的多样性知识。

- 采用数据集：LFW, AgeDB, and CPLFW。

- 网络模型：教师网络：ResNet-100,学生网络：[MobileFaceNet](https://arxiv.org/pdf/1804.07573.pdf)

- 算法思想：

  原始Triplet loss:
  $$
  \mathcal{L}=\frac{1}{N} \sum_{i}^{N} \max \left(\mathcal{D}\left(x_{i}^{a}, x_{i}^{p}\right)-\mathcal{D}\left(x_{i}^{a}, x_{i}^{n}\right)+m, 0\right)
  $$
  

​      在原始Triplet loss中，对于所有的identities m是相同的且固定不变的，所有的聚簇都将使用固定的距离粗鲁的分开，它忽视了identities之间微妙的相似性。比如说A和B的相似性大于A和C的相似性，那么理论上{A,B}的m应该小于{B,C}的m.和hinton的思想一样的，这样的相似性是有用的。
$$
\begin{array}{c}{\mathcal{L}=\frac{1}{N} \sum_{i}^{N} \max \left(\mathcal{D}\left(x_{i}^{a}, x_{i}^{p}\right)-\mathcal{D}\left(x_{i}^{a}, x_{i}^{n}\right)+\mathcal{F}(d), 0\right)} \\ {d=\max \left(\mathcal{T}\left(x_{i}^{a}, x_{i}^{n}\right)-\mathcal{T}\left(x_{i}^{a}, x_{i}^{p}\right), 0\right)}\end{array}
$$
Triplet Distillation:

先训练一个教师网络，然后教师网络提取identities的特征，计算它们之间的距离。如上所示$\mathcal{D}$代表学生网络计算的距离，$\mathcal{T}$代表教师网络的距离。

$\mathcal{F}$是一个简单的线性函数: $\mathcal{F}(d)=\frac{m_{\max }-m_{\min }}{d_{\max }} d+m_{\min }$

通过这种方式m被限制在了$m_{min}$ 和 $m_{max}$ 之间。