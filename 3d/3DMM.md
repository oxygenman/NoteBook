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
4. 求得形状和纹理协方差矩阵的特征值$\lambda_1$，$\lambda_2$和特征向量si，ti。

**转化后的模型为：**

$S_{m o d e l}=\bar{S}+\sum_{i=1}^{m-1} \lambda_{1i} s_{i}, T_{m o d e l}=\bar{T}+\sum_{i=1}^{m-1} \lambda_{2i} t_{i}$ 

### BFM模型

#### Model

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

$\alpha ,\beta$ 是需要学习的系数向量。

### FLAME 模型

FLAME构造了更加精确的和富于表情的头部和人脸模型，并且引入了头部姿势和眼球旋转。



![image-20220706144948951](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220706144948951.png)

#### <div align=center>![image-20220706145046054](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220706145046054.png) 

#### 模型细节

FLAME模型和SMPL模型一样，使用LBS。该模型设置了5023个顶点，4个关键点（脖子，下巴，和眼球），

$\begin{array}{l}M(\vec{\beta}, \vec{\theta}, \vec{\psi}): \mathbb{R}|\vec{\beta}| \times|\vec{\theta}| \times|\vec{\psi}| \rightarrow \mathbb{R}^{3 N}  \vec{\beta} \in \mathbb{R}^{|\vec{\beta}|}, \text { pose } \vec{\theta} \in \mathbb{R}^{|\vec{\theta}|},\vec{\psi} \in \mathbb{R}^{|\vec{\psi}|} \end{array}$ 

模型输入三种参数，就能得到5023个顶点坐标。

和SMPL模型一样，FLAME的组成部分有： template mesh, shape blend shape,pose blend shape,expression blend shape.

所以最终的模型是：

$\begin{array}{c}M(\vec{\beta}, \vec{\theta}, \vec{\psi})=W\left(T_{P}(\vec{\beta}, \vec{\theta}, \vec{\psi}), \mathbf{J}(\vec{\beta}), \vec{\theta}, \mathcal{W}\right) \\ \text { where } \\ T_{P}(\vec{\beta}, \vec{\theta}, \vec{\psi})=\overline{\mathbf{T}}+B_{S}(\vec{\beta} ; \mathcal{S})+B_{P}(\vec{\theta} ; \mathcal{P})+B_{E}(\vec{\psi} ; \mathcal{E})\end{array}$ 

这个模型怎么理解呢？

通过形状，姿态和表情参数以及一个均值模板可以得到一个特定人脸的静态3D模型，这个时候的人脸处在一个标准的姿态下。要想使头部的姿态发生变化，通过形状参数可以获得 0 pose,即一个标准姿态下人脸的关节点位置，然后再通过姿态参数$\theta$ 可以获取当前姿态下关节点的位置，然后再通过LBS,即当前的关节点位置×W,获得顶点的坐标。

#### (1)shape blendshapes

$\begin{array}{l}B_{S}(\vec{\beta} ; \mathcal{S})=\sum_{n=1}^{|\vec{\beta}|} \beta_{n} \mathbf{S}_{n} \\ \text { where } \vec{\beta}=\left[\beta_{1}, \cdots, \beta_{|\vec{\beta}|}\right]^{T} \text { denotes the shape coefficients, and } \\ \mathcal{S}=\left[\mathbf{S}_{1}, \cdots, \mathbf{S}_{|\vec{\beta}|}\right] \in \mathbb{R}^{3 N \times|\vec{\beta}|} \text { denotes the orthonormal shape basis, }  \end{array}$

####  (2)pose blendshapes

$B_{P}(\vec{\theta} ; \mathcal{P})=\sum_{n=1}^{9 K}\left(R_{n}(\vec{\theta})-R_{n}\left(\vec{\theta}^{*}\right)\right) \mathbf{P}_{n}$ 

其中，$R_n(\vec{\theta})$ ,表示将轴角向量转化为旋转矩阵。

$\begin{array}{l} \mathbf{P}_{n} \in \mathbb{R}^{3 N} \text { describes the vertex offsets from } \\ \text { the rest pose activated by } R_{n}, \text { and the pose space } \mathcal{P}=\left[\mathbf{P}_{1}, \cdots, \mathbf{P}_{9 K}\right] \in \mathbb{R}^{3 N \times 9 K} \end{array}$ 

包含所有的pose blend shapes.

$\mathcal{P}$可以看做一种形式的权重。

这里的$\mathcal{P}$ 是直接定义损失函数训练出来的。

#### (3)expression blendshapes

$\begin{array}{l}B_{E}(\vec{\psi} ; \mathcal{E})=\sum_{n=1}^{|\vec{\psi}|} \vec{\psi}_{n} \mathbf{E}_{n} \\ \text { where } \vec{\psi}=\left[\psi_{1}, \cdots, \psi_{|\vec{\psi}|^{T}}^{T}\right. \text { denotes the expression coefficients, and } \\ \mathcal{E}=\left[\mathbf{E}_{1}, \cdots, \mathbf{E}_{|\vec{\psi}|}\right] \in \mathbb{R}^{3 N \times|\vec{\psi}|} \text { denotes the orthonormal expression }\end{array}$

#### (4)Template shape:

从3D扫描数据集得到的平均模型。



2D->3D 驱动参数的学习

### DECA 模型

#### 加亿点点细节～

DECA主要关注于如何从2D图像恢复出逼真的3D人脸，所以它的主要内容是从2D图像中恢复出3DMM模型需要的参数及其他的一些细节内容。DECA不同于之前工作的主要内容是对皱纹如何跟随表情变化进行了建模，所以说是加入了一些细节，使生成的3D图形更加逼真。

#### 前置知识：

(1)Geometry prior:

本文用到的3D人头模型是FLAME，FLAME是一个统计学的模型，该模型输入三种参数：$\boldsymbol{\beta} \in \mathbb{R}^{|\boldsymbol{\beta}|}$ 表示shape参数或者叫identity参数，$\boldsymbol{\theta} \in \mathbb{R}^{3 k+3}$ 表示关节点参数，FLAME中有四个关节点两眼，下巴和脖子。$\boldsymbol\psi \in \mathbb{R}^{|\psi|}$ 表情参数。输出n=5023个vertices.模型可以表示为：

$M(\boldsymbol{\beta}, \boldsymbol{\theta}, \boldsymbol{\psi})=W\left(T_{P}(\boldsymbol{\beta}, \boldsymbol{\theta}, \boldsymbol{\psi}), \mathbf{J}(\boldsymbol{\beta}), \boldsymbol{\theta}, \boldsymbol{W}\right)$ 

W()是blend skining function,就是通过joint的位置和相应的权重W对顶点位置做一些变换。

其中：

$T_{P}(\boldsymbol{\beta}, \boldsymbol{\theta}, \boldsymbol{\psi})=\mathbf{T}+B_{S}(\boldsymbol{\beta} ; \mathcal{S})+B_{P}(\boldsymbol{\theta} ; \boldsymbol{P})+B_{E}(\boldsymbol{\psi} ; \mathcal{E})$

人头当前的形状，由人头模板加上三种blend shape组成，包括shape blend shape,pose blend shape,expression blend shape.

(2)Apperance model:表观模型，即皮肤的纹理颜色这些

本文用的是FLAME模型，但是FLAME模型没有表观模型，所以作者将BFM模型的albedo subspace转换到FLAME的uv layout.这个模型输入是$\boldsymbol{\alpha}\in\mathbb{R}^{|\alpha|}$ ,输出是UV alebedo map$A(\boldsymbol{\alpha}) \in \mathbb{R}^{d \times d \times 3}$ .

（3）camera model 

本文作者使用了一个正交相机模型，将3D mesh投影到了2d图像空间，映射关系为：

$\mathrm{v}=s \Pi\left(M_{i}\right)+\mathrm{t}$ 

其中$M_i$是3d顶点，$\Pi$ 是3d to 2d 的映射矩阵，s sacle,t是平移。

（4）Illumination model：

人脸领域最常用的光照模型是SH模型，该模型假设光源比较远，表面反射是Lambertian,即理想散射，那shaded image的计算公式是：

$B\left(\boldsymbol{\alpha}, \mathbf{l}, N_{u v}\right)_{i, j}=A(\boldsymbol{\alpha})_{i, j} \odot \sum_{k=1}^{9} \mathbf{l}_{k} H_{k}\left(N_{i, j}\right)$

A : albedo N:surface normal B:shaded texture

$H_{k}$ 表示SHbasis,$l_k$表示系数。

（5）texture rendering

Given the geometry parameters (𝜷, 𝜽, 𝝍), albedo (𝜶), lighting (l) and camera information 𝒄, we can generate the 2D image 𝐼𝑟 by rendering as 𝐼𝑟 = R (𝑀, 𝐵, c), where R denotes the rendering function

#### 方法

关键思想：

人脸会随着不同的表情变化，表现出不同的细节，但是他的一些固有的形状是不会变化的。

并且，人脸的细节信息应该被分成两种，一种是静态不变的个人细节，（比如痣，胡子，睫毛）和基于表情的细节（比如皱纹）。为了保持在表情变化引起的动态细节同时时保持静态细节，DECA学习了一个expression-conditional 细节模型，该模型能够产生出独立于表情的细节displacement map.个人理解将表情参数和人脸特征一同送入细节decoder模型，可以学习到一些不随表情变化的细节特征。

还有一个问题是，训练数据的获取比较困难，所以提出了一种直接从wild image学习几何细节的方法。

1.coarse recontruction 

![image-20220324004707658](/home/xy/pan/xy_workspace/git_workspace/notebook/3d/BFM.assets/image-20220324004707658.png)

粗糙重建指的是只学习FLAME模型的输入参数。如图所示，使用一个Encoder 模型直接回归出一些参数，(比如FLAME模型需要的参数（$\beta , \theta, \psi$）,反射率系数$\alpha$ ,相机参数$c$,光照参数$l$ ). 模型采用resnet50 模型，一共输出236维的latent code。并从重建的3d模型 投影出一张2d图片$I_r$和原来的图片进行对比，求一个损失。损失函数为：

$L_{\text {coarse }}=L_{l m k}+L_{e y e}+L_{p h o}+L_{i d}+L_{s c}+L_{r e g}$ 

关键点损失：2d ground truth和3d 重投影的损失：

$L_{l m k}=\sum_{i=1}^{68}\left\|\mathbf{k}_{i}-s \Pi\left(M_{i}\right)+\mathrm{t}\right\|_{1}$ 

闭眼损失：

$L_{e y e}=\sum_{(i, j) \in E}\left\|\mathbf{k}_{i}-\mathbf{k}_{j}-s \Pi\left(M_{i}-M_{j}\right)\right\|_{1}$ 

上眼皮关键点和下眼皮关键点距离的损失，这个损失可以减少3d和2d关键点没有对齐的影响。

图像本身的loss:

$L_{p h o}=\left\|V_{I} \odot\left(I-I_{r}\right)\right\|_{1,1}$ 

其中$V_I$ 表示脸部区域的mask ,通过脸部分割模型获得。

身份损失：

就是用一个特征提取网络，提取ground truth 图片和重投影图片的人脸特征，然后求一个余弦相似度。

$L_{i d}=1-\frac{f(I) f\left(I_{r}\right)}{\|f(I)\|_{2} \cdot\left\|f\left(I_{r}\right)\right\|_{2}}$ 

形状一致性损失：

给出一个人的两张不同照片Encoder $E_c$ 应该输出同样的参数，因为一个人的shape是不变的，变的是细节。

$L_{s c}=L_{\text {coarse }}\left(I_{i}, \mathcal{R}\left(M\left(\boldsymbol{\beta}_{j}, \boldsymbol{\theta}_{i}, \boldsymbol{\psi}_{i}\right), B\left(\boldsymbol{\alpha}_{i}, \mathbf{l}_{i}, N_{u v, i}\right), \mathbf{c}_{i}\right)\right)$ 

正则化项：

对需要学习的$\beta ,\psi, \alpha$ 进行L_2正则化。

2.细节重建

细节重建，使用一张细节UV偏移map,去增强FLAME的几何细节。和coarse重建一样，使用一个同样结构的Encoder,$E_d$ ,将输入图像编码到128维的latent code $\delta$ .然后再将这个latent code和FLAME的表情参数$\psi$ 和pose参数$\theta$ .拼接起来，通过$F_d$ 解码成D（UV displacement map). 为了渲染，D被转换为一个normal map.

细节渲染：

为了得到具有细节的M‘，我们将M和他的normal map，转化到UV 空间，

$M_{u v}^{\prime}=M_{u v}+D \odot N_{u v}$ 

其中D是detail code,$N_{uv}$ 代表normal map，$M_{uv}$ 应该是coarse model的UV map.

从M’ 计算得到N‘。然后就可以调用渲染函数进行渲染。B表示的是texture.

$I_{r}^{\prime}=\mathcal{R}\left(M, B\left(\boldsymbol{\alpha}, \mathbf{1}, N^{\prime}\right), \mathbf{c}\right)$ 

从而可以得到渲染后的图片$I_r ’$ .

$L_{\text {detail }}=L_{p h o D}+L_{m r f}+L_{s y m}+L_{d c}+L_{r e g D}$ .

ID-MRF loss:

[ID-MRF](https://blog.csdn.net/qq_37937847/article/details/117163628)

隐式多元马尔科夫随机场损失.用来惩罚生成图像中的每个patch只和target中大部分的patch比较相似的情况，所以能够恢复出细节。

要计算ID-MRF损失，可以简单地使用直接相似度度量(如余弦相似度)来找到生成内容中的补丁的最近邻居。但这一过程往往产生平滑的结构，因为一个平坦的区域容易连接到类似的模式，并迅速减少结构的多样性。我们采用相对距离度量[17,16,22]来建模局部特征与目标特征集之间的关系。它可以恢复如图3(b)所示的细微细节。
![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20210522160236955.png)

具体地，用$Y g ∗ $ 代表对缺失区域的修复结果的内容，$ Y_g^{*L}$和 $Y^L$ 分别代表来自预训练模型的第L层的特征。

patch v和s分别来自$ Y_g^{*L}$和$ Y^L$ ,定义v与s的相对相似度为：

$\operatorname{RS}(\mathbf{v}, \mathbf{s})=\exp \left(\left(\frac{\mu(\mathbf{v}, \mathbf{s})}{\max _{\mathbf{r} \in \rho_{\mathbf{s}}\left(\mathbf{Y}^{L}\right)} \mu(\mathbf{v}, \mathbf{r})+\epsilon}\right) / h\right)$ 

这里$\mu()$ 是计算余弦相似度。$r\in\rho_s(Y^L)$ 意思是r是$Y^L$ 中除了sd的其他patch.h和$\epsilon$ 是两个超参数常数。仔细观察这个相对相似度和原始相似度的关系，会发现如果最高的相似度作为分母的话，那相对相似度就会变小，也就是小的更小，大的更大。接下来：RS(v,s)归一化为：

$\overline{\mathrm{RS}}(\mathbf{v}, \mathbf{s})=\operatorname{RS}(\mathbf{v}, \mathbf{s}) / \sum_{\mathbf{r} \in \rho_{\mathbf{s}}\left(\mathbf{Y}^{L}\right)} \mathrm{RS}(\mathbf{v}, \mathbf{r})$

最后，根据上式，最终的ID-MRF损失被定义为：

$\mathcal{L}_{M}(L)=-\log \left(\frac{1}{Z} \sum_{\mathbf{s} \in \mathbf{Y}^{L}} \max _{\mathbf{v} \in \hat{\mathbf{Y}}_{g}^{L}} \overline{\mathrm{RS}}(\mathbf{v}, \mathbf{s})\right)$

一个极端的例子$Y_g^{*L}$ 中的所有patch都非常接近目标中的一个patch s.而对于其他的patch r $max_vRS(v,r)$ 就会变小。$L_m$ 就会变大。

另一方面，$Y^L$中的每一个patch r 在$Y_g^{*L}$ 中有一个唯一的最近邻。那么结果就是RS(v,r)变大。$L_m$就会变小。

从这个观点触发，最小化，LM(L)鼓励$Y_g*^{L}$ 中的每一个patch v都匹配Y^L中不同的patch.是的变得多样化。
$$
L_{m r f}=2 L_{M}\left({ conv4_2) }+L_{M}({ conv3_2 })\right.
$$
Soft symmetry loss：

对称损失，增加遮挡的损失。

$L_{s y m}=\left\|V_{u v} \odot(D-f l i p(D))\right\|_{1,1}$ 

正则化损失：

$L_{r e g D}=\|D\|_{1,1}$ 

4.3细节解耦

核心的依据是，同一个人的不同照片，除了表情控制的细节。其他的细节和大致的形状是不变的。

交换同一个人两张照片的detail code , 不会影响照片的三维重建，也就是说他们的detail code 应该是相同的。

所以构造了如下损失函数：

Detail consistency loss: 
$$
\begin{array}{r}L_{d c}=L_{d e t a i l}\left(I_{i}, \mathcal{R}\left(M\left(\boldsymbol{\beta}_{i}, \boldsymbol{\theta}_{i}, \boldsymbol{\psi}_{i}\right), A\left(\boldsymbol{\alpha}_{i}\right)\right.\right. \\ \left.\left.F_{d}\left(\boldsymbol{\delta}_{j}, \boldsymbol{\psi}_{i}, \boldsymbol{\theta}_{j a w, i}\right), \mathbf{l}_{i}, \mathbf{c}_{i}\right)\right)\end{array}
$$
给出一个人两张不同的照片$I_i$ 和 $I_j$ .损失函数如上所示。其中$\delta_j$ 表示$I_j$ 的detail code .

![image-20220706192231323](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220706192231323.png)

$L_{dc}$ 对模型的影响。







