## MnasNet: Platform-Aware Neural Architecture Search for Mobile

### 摘要

MnasNet将模型在手机上的latency引入到NAS中，实现了accuray和latency的平衡。并引入了factorized hierarchical

search space，可以实现层间多样性，而不像之前每层都是相同结构的重复。

实验结果：

搜索一个网络需要4.5days on 64TPUv2.

| *imagenet*/pixelphone | latency   | accuracy   |
| --------------------- | --------- | ---------- |
| MnasNet               | 78ms      | 75.2%top-1 |
| mobileNet-v2          | 1.8x less | 0.5% less  |
| NasNet                | 2.3x less | 1.2% less  |

[github开源地址](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)

### 问题公式化

为了在一个网络架构搜索中实现普雷托最优，（ps:普雷托最优：从此以后，非损人不能利己-知乎），就是说要么实现最高的精度，而不增加时延；要么实现最低的时延而不降低精度。论文使用了一种customized weighted product方式来近似pareto-optimal solutions.
$$
\underset{m}{\operatorname{maximize}} \quad A C C(m) \times\left[\frac{L A T(m)}{T}\right]^{w}
 \\where \quad
  w=\left\{\begin{array}{ll}\alpha, & \text { if } L A T(m) \leq T \\ \beta, & \text { otherwise }\end{array}\right.
$$
ACC(m)表示精度，LAT(m)表示时延，T表示目标时延。

根据经验求得$\alpha$和$\beta$的值：根据经验时延翻倍，精度增加5%，所以代入上述公式
$$
\begin{array}{l} \operatorname{Re} \text {ward}(M 2)=a \cdot(1+  5 \%) \cdot(2 l / T)^{\beta} \approx \operatorname{Re} \text {ward}(M 1)=a \cdot(l / T)^{\beta} \end{array}
$$
算得$\alpha=\beta=-0.07$ 。所以这就是我们的目标函数。

### Mobile Neural Architecture Search

1. Factorized Hierarchical Search Space (分层搜索空间)

   之前工作的搜索都是在几个cells上进行搜索，然后不断的进行堆叠，丧失了层间的多样性。

   所以本文将整个CNN model划分成多个block ,每个block是一个相对独立的搜索空间，这样做可以降低搜索的复杂度，并且增加多样性，每个搜索空间可以搜索的操作：

   - Convolotional ops: regular conv, depthwise conv, mobile inverted bottleneck conv.
   - Convolutional kernel size :3x3, 5x5
   - Squeeze-and-excitation ratio SERatio:0, 0.25.
   - Skips ops:pooling,idetity residual, no skip.
   - Output filter size $F_i$
   - Number of layers per block $N_i$     

   本文的搜索空间都是以MoblileNetV2为参照物进行离散化的，比如在某个block中，某个layer的有无通过{0,+1,-1}来标识。filtersize通过相对MobileNet的比例{0.75, 1.0,1.25}来标识。

2. 搜索算法

   强化学习

   