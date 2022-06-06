## 								FLAME 3D 人脸模型

### FLAME模型的主要贡献

FLAME构造了更加精确的和富于表情的头部和人脸模型，并且引入了头部姿势和眼球旋转。并且开源了FLAME trained 模型，开源了基于D3DFACS的对齐时序人脸头部数据集。

### 模型综述

FLAM模型和SMPL模型一样，使用LBS。该模型设置了5023个顶点，4个关键点（脖子，下巴，和眼球），

$\begin{array}{l}M(\vec{\beta}, \vec{\theta}, \vec{\psi}): \mathbb{R}|\vec{\beta}| \times|\vec{\theta}| \times|\vec{\psi}| \rightarrow \mathbb{R}^{3 N}  \vec{\beta} \in \mathbb{R}^{|\vec{\beta}|}, \text { pose } \vec{\theta} \in \mathbb{R}^{|\vec{\theta}|},\vec{\psi} \in \mathbb{R}^{|\vec{\psi}|} \end{array}$ 

模型输入三种参数，就能得到5023个顶点坐标。

和SMPL模型一样，FLAM的组成部分有： template mesh, shape blend shape,pose blend shape,expression blend shape.

所以最终的模型是：

$\begin{array}{c}M(\vec{\beta}, \vec{\theta}, \vec{\psi})=W\left(T_{P}(\vec{\beta}, \vec{\theta}, \vec{\psi}), \mathbf{J}(\vec{\beta}), \vec{\theta}, \mathcal{W}\right) \\ \text { where } \\ T_{P}(\vec{\beta}, \vec{\theta}, \vec{\psi})=\overline{\mathbf{T}}+B_{S}(\vec{\beta} ; \mathcal{S})+B_{P}(\vec{\theta} ; \mathcal{P})+B_{E}(\vec{\psi} ; \mathcal{E})\end{array}$

#### (1)shape blendshapes

$\begin{array}{l}B_{S}(\vec{\beta} ; \mathcal{S})=\sum_{n=1}^{|\vec{\beta}|} \beta_{n} \mathbf{S}_{n} \\ \text { where } \vec{\beta}=\left[\beta_{1}, \cdots, \beta_{|\vec{\beta}|}\right]^{T} \text { denotes the shape coefficients, and } \\ \mathcal{S}=\left[\mathbf{S}_{1}, \cdots, \mathbf{S}_{|\vec{\beta}|}\right] \in \mathbb{R}^{3 N \times|\vec{\beta}|} \text { denotes the orthonormal shape basis, }  \end{array}$

####  (2)pose blendshapes

$B_{P}(\vec{\theta} ; \mathcal{P})=\sum_{n=1}^{9 K}\left(R_{n}(\vec{\theta})-R_{n}\left(\vec{\theta}^{*}\right)\right) \mathbf{P}_{n}$ 

其中，$R_n(\vec{\theta})$ ,表示将轴角向量转化为旋转矩阵。

$\begin{array}{l} \mathbf{P}_{n} \in \mathbb{R}^{3 N} \text { describes the vertex offsets from } \\ \text { the rest pose activated by } R_{n}, \text { and the pose space } \mathcal{P}=\left[\mathbf{P}_{1}, \cdots, \mathbf{P}_{9 K}\right] \in \mathbb{R}^{3 N \times 9 K} \end{array}$ 

包含所有的pose blend shapes.

这里的$\mathcal{P}$ 是直接定义损失函数训练出来的。

#### (3)expression blendshapes

$\begin{array}{l}B_{E}(\vec{\psi} ; \mathcal{E})=\sum_{n=1}^{|\vec{\psi}|} \vec{\psi}_{n} \mathbf{E}_{n} \\ \text { where } \vec{\psi}=\left[\psi_{1}, \cdots, \psi_{|\vec{\psi}|^{T}}^{T}\right. \text { denotes the expression coefficients, and } \\ \mathcal{E}=\left[\mathbf{E}_{1}, \cdots, \mathbf{E}_{|\vec{\psi}|}\right] \in \mathbb{R}^{3 N \times|\vec{\psi}|} \text { denotes the orthonormal expression }\end{array}$

#### (4)Template shape:

从3D扫描数据集中学到的。

### Temporal Registration

registration的过程和FLAME模型的训练和regularing交替进行。

#### （1）初始化模型

要进行上述过程，我们首先要初始化一个FLAME模型。$\begin{array}{l}\text { As described in Section } 3 \text {, FLAME consists of parameters for shape } \\ \{\overline{\mathbf{T}}, \mathcal{S}\} \text {, pose }\{\mathcal{P}, \mathcal{W}, \mathcal{J}\} \text {, and expression } \mathcal{E} \text {, that require an initial- } \\ \text { ization, which we then refine to fit registered scan data. }\end{array}$

 shape的初始化：

基于CAESAR数据集,extract 并refine了SMPL的registration的头部区域,调整了模型拓扑，使其拥有眼部和嘴部的空洞，为了增加模型的视觉品质，添加了眼球模型Woods et al. [2016]。

pose:

blend weights $\mathcal W$ 和 joint regressor $\mathcal{J}$ 是人工定义的，眼球的joint regerssor手工的让回归结果在眼球的中心。

Expression:

我们建立了我们的head template和人工的FACS-based blend shape model的对应关系，然后使用deformation transfer，将expression blend shape transfer到我们的模型上,然后算一个和template之间的差值，作为初始值（个人理解).

#### （2）Single-frame registration

我们要对齐的数据包含三部分，3D SCAN vertices,多角度视图，相机参数。

这个过程分为3步：

（1）model only

首先优化模型系数$\{\vec{\beta},\vec{\theta},\vec{\psi}\}$， 损失函数为：

$\begin{array}{l}\qquad E(\vec{\beta}, \vec{\theta}, \vec{\psi})=E_{D}+\lambda_{L} E_{L}+E_{P} \\ \text { with the data term } \\ \qquad E_{D}=\lambda_{D} \sum_{\mathbf{v}_{s}} \rho\left(\min _{\mathbf{v}_{m} \in M(\vec{\beta}, \vec{\theta}, \vec{\psi})}\left\|\mathbf{v}_{s}-\mathbf{v}_{m}\right\|\right)\end{array}$ 

其中$E_L$是关键点损失，关键点是已知多角度视图和模型利用相机参数投影后的关键点之间的L2损失，$E_P$ 是正则化项。

（2）model和对齐联合优化

需要优化的损失函数为：

这里的$E_D$ 表示model和对齐的template之间的损失。

$E(\mathbf{T}, \vec{\beta}, \vec{\theta}, \vec{\psi})=E_{D}+E_{C}+E_{R}+E_{P}$ 

其中

$E_{C}=\sum_{e} \lambda_{e}\left\|\mathbf{T}_{e}-M(\vec{\beta}, \vec{\theta}, \vec{\psi})_{e}\right\|$ 

表示template和模型的edge之间的损失。

正则项：

$E_{R}=\frac{1}{N} \sum_{k=1}^{N} \lambda_{k}\left\|U\left(\mathbf{v}_{k}\right)\right\|^{2}$ 

表示顶点间的离散拉普拉斯近似。

$U(\mathbf{v})=\frac{\sum_{\mathbf{v}_{r} \in \mathcal{N}(\mathbf{v})} \mathbf{v}_{r}-\mathbf{v}}{|\mathcal{N}(\mathbf{v})|}$ 

正则项让registration避免重叠，使registration对噪声和遮挡更加鲁棒。

（3）基于texture的优化

$E(\mathbf{T}, \vec{\beta}, \vec{\theta}, \vec{\psi})=E_{D}+E_{C}+\lambda_{T} E_{T}+E_{R}+E_{P}$ 

其中，

$E_{T}=\sum_{l=0}^{3} \sum_{v=1}^{V}\left\|\Gamma\left(I_{l}^{(v)}\right)-\Gamma\left(\hat{I}_{l}^{(v)}\right)\right\|_{F}^{2}$ 

表示真实图像和渲染模型投影图像之间的差异。

### （3）Sequential registration

（1）personalization

就是每一个不同的人使用不同的模板。通过co-reigistration得到。

（2）Sequence fitting

使用personaliza的模板去替代平均模板，并且将形状参数$\beta$置0.使用上一帧的模型初始化下一帧的模型，调用单帧拟合过程，生成新的FLAME模型，再重复上述所有过程。

### DATA

### MODEL TRAINING

获得对齐后的数据后，训练的目标是将pose,shape,expresison进行解耦训练。我理解训练过程已经在对齐过程中描述过了，这里就不看了。

1.pose parameter training

模型包含两种类型的参数。

一种是基于个人的templates $T_i^P$ 和$J_i^P$ .

还有一种是全局参数：blend weights $\mathcal{W}$ ,pose blend shapes$\mathcal{P}$ ,joint regressor$\mathcal{J}$ .

优化过程在个人参数和全局参数之间交替进行。

2. Expression parameter training

   首先去除pose和shape的影响，然后使用PCA算法，去求expression的主成分。

3. shape parameter traing

   去除pose和expression的影响，然后是用PCA算法。

   





