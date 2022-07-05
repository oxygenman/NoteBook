[人脸3D模型的发展](https://zhuanlan.zhihu.com/p/161828142) 

[罗德里格斯公式推导](https://zhuanlan.zhihu.com/p/113299607)



### 基于图像的人脸三维重建方法

1. 立体匹配（Structure From Motion,SfM)
2. Shape from Shading,sfs
3. ***三维可变形人脸模型（3DMM)***

### 什么是3DMM模型

3DMM,即三维可变形人脸模型，它可以使用固定数量的参数来表示一个三维人脸。

**核心思想：一个三维人脸可以可以由其他许多幅人脸正交基加权线性相加而来。**

类比我们所处三维空间中的点，每一点(x,y,z),实际上都由三维空间三个方向的基量，（1,0,0），（0,1,0），（0,0,1）加权相加所得，权重分别是x,y,z.

那么对于人脸来说，一个人脸可以由其他多幅人脸加权相加得到。在BFM模型中，将人脸的表示，分为形状向量和纹理向量，即一个人脸分为形状和纹理两部分叠加。如图所示：

人脸的形状可以表示为一个向量Shape Vector: $S=(x_1,y_1,z_1,x_2,y_2,z_2,...,x_n,y_n,z_n)$ 即人脸表面点的三维坐标。

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-58bb650a63e71c194ff00469753cf6b6_720w.jpg)

纹理向量Texture Vector:$T=(r_1,g_1,b_1,r_2,g_2,b_2,...,r_n,g_n,b_n)$ ,即每个点的的颜色信息。

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-6a90450d84d1fb00cc7891c042fad17f_720w.jpg)

任意的人脸模型可以由数据集中的m个人脸模型进行加权组合如下：

$\mathbf{S}_{\text {mod }}=\sum_{i=1}^{m} a_{i} \mathbf{S}_{i}, \quad \mathbf{T}_{\text {mod }}=\sum_{i=1}^{m} b_{i} \mathbf{T}_{i}, \quad \sum_{i=1}^{m} a_{i}=\sum_{i=1}^{m} b_{i}=1$ 

其中$S_i$ 和 $T_i$ 就是数据库中的第i张人脸的形状向量和纹理向量。

**但是在实际构建模型的时候不能使用$S_i$ 和 $T_i$ 作为基向量，因为它们之间不是正交的。**



使用PCA进行降维分解，求正交基

1. 首先计算形状和纹理向量的平均值。

2. 中心化人脸数据。

3. 分别计算协方差矩阵

4. 求得形状和纹理协方差矩阵的特征值α，β和特征向量si，ti。

   **转化后的模型为：**

   $S_{m o d e l}=\bar{S}+\sum_{i=1}^{m-1} \alpha_{i} s_{i}, T_{m o d e l}=\bar{T}+\sum_{i=1}^{m-1} \beta_{i} t_{i}$

   



### BFM模型

### Model

$\begin{aligned} \mathbf{s} &=\left(x_{1}, y_{1}, z_{1}, \ldots x_{m}, y_{m}, z_{m}\right)^{T} \\ \mathbf{t} &=\left(r_{1}, g_{1}, b_{1}, \ldots r_{m}, g_{m}, b_{m}\right)^{T} \end{aligned}$

一个人脸使用两个向量表示，顶点坐标$ \left(x_{j}, y_{j}, z_{j}\right)^{T} \in \mathbb{R}^{3} , 顶点颜色$ $\left(r_{j}, g_{j}, b_{j}\right)^{T} \in[0,1]^{3} $ 。m=53490个顶点。

BFM假定形状和纹理是相互独立的。

使用数据集构建一个高斯模型：

$\mathcal{M}_{s}=\left(\boldsymbol{\mu}_{s}, \boldsymbol{\sigma}_{s}, \mathbf{U}_{s}\right) \text { and } \mathcal{M}_{t}=\left(\boldsymbol{\mu}_{t}, \boldsymbol{\sigma}_{t}, \mathbf{U}_{t}\right)$ 

其中均值：$\boldsymbol{\mu}_{\{s, t\}} \in \mathbb{R}^{3 m}$ ,

​	标准差：$\boldsymbol{\sigma}_{\{s, t\}} \in \mathbb{R}^{n-1}$ 

​	正交基：$ \mathbf{U}_{\{s, t\}}=\left[\mathbf{u}_{1}, \ldots \mathbf{u}_{n}\right] \in  \mathbb{R}^{3 m \times n-1}$ 

值得注意的是正交基的长度为1，所以乘 $diag(\sigma_{s})$ 相当于将U还原到协方差的量级。   

那么最后的模型为：

$\begin{array}{l}\mathbf{s}(\alpha)=\boldsymbol{\mu}_{s}+\mathbf{U}_{s} \operatorname{diag}\left(\boldsymbol{\sigma}_{s}\right) \alpha \\ \mathbf{t}(\beta)=\boldsymbol{\mu}_{t}+\mathbf{U}_{t} \operatorname{diag}\left(\boldsymbol{\sigma}_{t}\right) \beta\end{array}$ 

$\alpha ,\beta$ 应该是主成分系数

#### 





