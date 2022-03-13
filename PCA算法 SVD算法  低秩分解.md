## PCA算法 SVD算法  低秩分解

1.什么是相似矩阵？

同一个线性变换，不同基下的矩阵，称为相似矩阵。

https://blog.csdn.net/Dark_Scope/article/details/53150883

2.[PCA知乎讲解](https://zhuanlan.zhihu.com/p/32412043)

3.[PCAB站讲解](https://www.bilibili.com/video/BV1E5411E71z?from=search&seid=1616609056191763735&spm_id_from=333.337.0.0)

[SVD分解](https://www.bilibili.com/video/BV16A411T7zX?spm_id_from=333.999.0.0)

PCA

1.目标：找到一个新的坐标系，使数据集在新的坐标系下方差最大

2.我们的数据是由白数据左乘拉伸矩阵S，再左乘旋转矩阵R的到的。那么拉伸的方向就是方差最大的方向，那我们的目标就是要找 这个旋转矩阵。

![image-20211219164843861](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211219164843861.png)

旋转矩阵$R^{-1}=R^T$ 

怎么求R呢？

协方差矩阵的特征向量就是R

 协方差是什么？

协方差表示两个向量是同乡变化和是反向变化，以及变化的程度如何。

SVD分解

svd分解的物理意义 M代表一种线性变化，一组正交基，经过M的线性变化后依然垂直，只不过长度进行了缩放，方向发生了变化。

$MV=U\Sigma$   V为原始正交基，$\Sigma$ 为变换后的空间的正交基
