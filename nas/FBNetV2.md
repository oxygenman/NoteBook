##  Differentiable Neural Architecture Search for Spatial and channel Dimensions

### 摘要

本文主要贡献是在基本不增加计算量和内存占用的前提下，在搜索空间中增加了对resolution和filter数量的搜索，比传统DNAS（Differentiable Neural Architecture Search)的搜索空间增加了$10^{14}$ 倍。该效果主要通过一种掩码机制来实现，作者称为DMaskingNAS.DmaskingNAS 取得了比MobilNetV3-small高0.9%的准确率并且减少了15%的FLOPS。在ImageNet分类任务上with only 27 hours on 8 GPUS.

|                   | accuracy | FLOPS |
| ----------------- | -------- | ----- |
| FBNetV2           | VS       | VS    |
| MoblieNetV3-small | +0.9%    | -15%  |
| MoblieNet         | +2.6%    | =     |
| Efficient-B0      | =        | -20%  |



### 方法

1. Channel Search

   <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gez11797kdj30tu0s4gnh.jpg" alt="Channel_search.jpg" style="zoom: 33%;" />

   step A: 在同一个cell中，不同block可能维度不同，无法直接相加，

   step B: 可以通过补0的方式，把不同维度的channel补齐，

   step C: step B等同于使用一个0，1mask和相同维度的filters相乘，

   step D: 使用weight sharing的方式将三个filter归为1个相同的filter,

   step E: 等同于先将三个mask相加，再和filter相乘。

   最终的形式是：

   $y=b(x) \circ \underbrace{\sum_{i=1}^{k} g_{i} \mathbb{1}_{i}}_{M}$

   其中$b(x)$ 为block,$g_i$ 为Gumbel softmax的输出，$\mathbb{1}_{i}$ 为mask.

   

2. Input Resolution Search

   不同的输入分辨率如果直接padding后相加，可能造成像素级别的无法对齐，和感受野的不匹配。

   

   <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gez2gmo3l1j30iq0oa40e.jpg" alt="image-20200520175658071" style="zoom:50%;" />

A：不同的分辨率无法直接相加

B： 如果简单补零会造成像素的不匹配

C： 可以同过内部插0的方式来对齐

D：如果只接卷积，感受野会不同

E； 可以通过先卷积再插0的方式解决

