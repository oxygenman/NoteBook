[人脸3D模型的发展](https://zhuanlan.zhihu.com/p/161828142) 

[罗德里格斯公式推导](https://zhuanlan.zhihu.com/p/113299607)

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

### Experiments 

#### 3.1 人脸识别

#### 3.2 3D scans的识别





