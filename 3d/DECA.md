DECA模型

## 1.Introduction:

现有大部分的单目3d人脸重建算法可以很好的重建出人脸的几何特征，但是也存在一些缺点。比如不能很好的进行自然的动画控制，因为它们没有对皱纹如何跟随表情变化进行建模；一些模型是在高清的扫描数据集上进行训练的，无法对wild image进行泛华。所以作者提出了DECA(Detailed Expression Capture and Animation).该模型可以从一个低维的表征（包括detail参数和表情参数）回归出一个UV displacement map，同时还有一个回归模型可以从一张2d图片回归出detail,shape,albedo,expression,pose和illumimation参数。为了实现这个模型，作者提出了一个detail-consistency loss可以将表情导致的皱纹和本有的皱纹细节分开。这样就可以在控制表情变化的同时而不影响原有的细节，是重建更加自然。值得一提的是DECA使用的训练数据全部是2d人脸，而没有使用3d或4d扫描数据。

### 3.Preliminaries:

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

4.Method 

关键思想：

人脸会随着不同的表情变化，表现出不同的细节，但是他的一些固有的形状是不会变化的。

并且，人脸的细节信息应该被分成两种，一种是静态不变的个人细节，（比如痣，胡子，睫毛）和基于表情的细节（比如皱纹）。为了保持在表情变化引起的动态细节同时时保持静态细节，DECA学习了一个expression-conditional 细节模型，该模型能够产生出独立于表情的细节displacement map.个人理解将表情参数和人脸特征一同送入细节decoder模型，可以学习到一些不随表情变化的细节特征。

还有一个问题是，训练数据的获取比较困难，所以提出了一种直接从wild image学习几何细节的方法。

4.1 coarse recontruction 

![image-20220324004707658](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220324004707658.png)

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

4.2 细节重建

细节重建，使用一张细节UV偏移map,去增强FLAME的几何细节。和coarse重建一样，使用一个同样结构的Encoder,$E_d$ ,将输入图像编码到128维的latent code $\delta$ .然后再将这个latent code和FLAME的表情参数$\psi$ 和pose参数$\theta$ .拼接起来，通过$F_d$ 解码成D（UV displacement map). 为了渲染，D被转换为一个normal map.

细节渲染：

为了得到具有细节的M‘，我们将M和他的normal map，转化的UV 空间，

$M_{u v}^{\prime}=M_{u v}+D \odot N_{u v}$ 

其中D是detail code,$N_{uv}$ 代表normal map，$M_{uv}$ 应该是coarse model的UV map.

从M’ 计算得到N‘。然后就可以调用渲染函数进行渲染。B表示的是texture.

$I_{r}^{\prime}=\mathcal{R}\left(M, B\left(\boldsymbol{\alpha}, \mathbf{1}, N^{\prime}\right), \mathbf{c}\right)$ 

从而可以得到渲染后的图片$I_r ’$ .

$L_{\text {detail }}=L_{p h o D}+L_{m r f}+L_{s y m}+L_{d c}+L_{r e g D}$ .

ID-MRF loss:

[ID-MRF](https://blog.csdn.net/qq_37937847/article/details/117163628)

隐式多元马尔科夫随机场损失.用来惩罚生成图像中的每个patch只和target中大部分的patch比较相似的情况，所以能够恢复出细节。

![image-20220519172402857](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220519172402857.png)
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

<img src="/Users/xy/Library/Application Support/typora-user-images/image-20220324002617847.png" alt="image-20220324002617847" style="zoom: 67%;" />

$L_{dc}$ 对模型的影响。

相关知识:

#### Texture Space:

FLAME Texture Space的获取过程：

(1)首先为了获得一个初始化的texture space ,先将FLAME模型拟合到BFM模型上，并将BFM的vertex投影到FLAM上，以此获得一个初始化的texture space.

(2)然后将FLAM模型拟合到FFHQ数据集的图片上，（使用deca或者其他的方法）。并获得每张图片的texture offset.

(3)使用一个图像补全网络（GMCNN），补全被遮挡的texture map.

(4)的到1500张textue mapl。使用PCA算法获得一个textue space.

#### Render :

[[Rendering pipeline 之　Rasterizer](http://www.cppblog.com/lijinshui/archive/2008/12/02/68367.html)](http://www.cppblog.com/lijinshui/archive/2008/12/02/68367.aspx)

[3d渲染过程](https://blog.csdn.net/qq_40822303/article/details/86664774)

#### UV 贴图：

[uv贴图类型](https://www.bbsmax.com/A/q4zVEDQ7dK/)

[理解UV贴图](https://www.bbsmax.com/A/gGdXqgPQz4/)
