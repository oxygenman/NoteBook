

[三维数字人体模型](https://zhuanlan.zhihu.com/p/53072321)

[rigging,skkinning的意思](https://blog.csdn.net/linjf520/article/details/89951103)

[SMPL: A Skinned Multi-Person Linear Model论文解读](https://blog.csdn.net/JerryZhang__/article/details/103478265)

[单目实时全身动作捕捉](https://zhuanlan.zhihu.com/p/374389244)

### SMPL(skinned multi-person linear model)

SMPL模型是一种参数化人体模型，是马普所提出的一种人体建模方法，该方法可以进行任意的人体建模和动画驱动。

smpl是一种skinned的，基于顶点（vertex-based）的人体三维模型，能够精确地表示人体的不同形状（shape）和姿态（pose）。

smpl适用于动画领域，可以随姿态变化自然的变形，并伴随软组织的自然运动。

smpl是一种可学习的模型，通过训练可以更好的拟合人体的形状和不同姿态下的形变。

模型学习使用到的数据：静息姿态模板，混合权重，不同姿态的混合形状，不同个体的混合形状。

Specifically we learn blend shapes to correct for the limitations of standard skinning. Different blend shapes for identity, pose, and soft-tissue dynamics are additively combined with a rest template before being transformed by blend skinning. 

 A key component of our approach is that we formulate the pose blend shapes as a linear function of the elements of the part rotation matrices

#### 相关概念：

Blend Skinning . skeleton subspace deformation methods, also known as blend skinning, attach the surface of a mesh to an ubderlying skeletal structure.

骨架子空间变形方法，将mesh上的点与骨架进行绑定，mesh上的每个vertex都以一定的权重和骨架相连，所以使用骨架可以控制mesh的形变，vetex受到与他相邻的估计的加权影响，这种影响可以通过LBS来实现。

Auto-rigging:

自动的生成LBS的权重，否则需要人工绑定。take a collection of meshes and infer the bones as well as the joints and blend weights 。

Blend shapes:

which defines the corrections in a rest pose and then applies a standard skinning equation (e.g. LBS).先对rest pose 进行修正，然后再使用标准的skinning方法，（比如LBS）.

The idea is to define corrective shapes (sculpts) for specific key poses, so that when added to the base shape and transformed by blend skinning, produce the right shape.

为特定的关键pose定义修正形状（雕刻）,并添加到base shape,使用blend skinning进行变形，最终产生正确的形状。这一过程之前通常是人工来实现的，非常的繁琐，耗时。

A blend shape is a vector of vertex displacements in a rest pose

#### 整体建模：

![image-20210714160543263](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210714160543263.png)

和SCAPE一样，将body shape按身份和非刚性姿态进行划分。

一个单独的blend shape使用一个vertex 偏移量组成的向量表示。

基本的artist-created mesh由6890个定点和23个链接点构成，如图6所示：

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210714164141323.png" alt="image-20210714164141323" style="zoom:50%;" />



模型由以下参数和函数定义而成：

 **mean template shape:** 在zero pose$\vec{\theta}^{*}$ ，状态下$\overline{\mathbf{T}} \in \mathbb{R}^{3 N}$ 

**blend weights:** $W\in\mathbb{R}^{N*K}$ K是连接点个数。如figure3(a)所示。

**blend shape function:** $B_{S}(\vec{\beta}): \mathbb{R}^{|\vec{\beta}|} \mapsto \mathbb{R}^{3 N}$ ，输入是形状参数向量$\vec{\beta}$ ,输出blend shape.

**预测K个链接点的function：**$J(\vec{\beta}): \mathbb{R}^{|\vec{\beta}|} \mapsto \mathbb{R}^{3K}$ ,输入依然是形状参数向量，输出是连接点的坐标。

**基于pose的blend shape function:**$B_{P}(\vec{\theta}): \mathbb{R}^{|\vec{\theta}|} \mapsto \mathbb{R}^{3 N}$ ,

输入是一个pose parameter $\vec{\theta}$ ,这种修正的blend shape 和 静息装填的blend shap加在一起构成新的blend shape.

**标准的blend skinning函数**：$W(.)$ ,可以是LBS ，也可以是dual-quaternion. 

**最终的model:**$M(\vec{\beta}, \vec{\theta} ; \Phi): \mathbb{R}^{|\vec{\theta}| \times|\vec{\beta}|} \mapsto \mathbb{R}^{3 N}$ 

整个过程可以参考fig3.所以整个模型的输入是形状参数向量$\vec{\beta}$ 和 姿态参数向量 $\vec{\theta}$ .输出的是blend shape.

### 详细分解：

#### Blend skinning:

本文使用LBS.

身体姿态使用标准的skeletal rig 定义：

$\vec{w_K}\in \mathbb{R}^3$ 表示身体部位k和它的在kinematic tree上的父节点的相对轴角。

[axis-angel representation 定义详见维基百科](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)

所以上面提到的姿态参数向量是$\vec{\theta}=[\vec{w_0}^T,...,\vec{w_K}^T]^T$

K表示23个连接点，所以$|\vec{\theta}|=3*23+3=72$ 个参数，+3表示根方向。

根据轴角的定义，$\overline{w}=\frac{\vec{w}}{||\vec{w}||}$ 表示旋转轴，$||\vec{w}||$ 表示旋转角度。

根据罗德里格斯公式，算出3d旋转矩阵：

$\exp \left(\vec{\omega}_{j}\right)=\mathcal{I}+\widehat{\overline\omega}_{j} \sin \left(\left\|\vec{\omega}_{j}\right\|\right)+\widehat{\overline\omega}_{j}^{2} \cos \left(\left\|\vec{\omega}_{j}\right\|\right)$

可以得到一个3*3的反对称旋转矩阵$\widehat{\overline\omega_j}$.

通过这个矩阵可以计算出新的关键点的位置。然后再使用$W(.)$ 函数生成新的定点vertices坐标。$W(\overline{\mathbf{T}}, \mathbf{J}, \vec{\theta}, \mathcal{W}): \mathbb{R}^{3 N \times 3 K \times|\vec{\theta}| \times|\mathcal{W}|} \mapsto \mathbb{R}^{3 N}$，输入是rest pose,$\overline{T}$,连接点位置，$J$, 姿态，$\vec{\theta}$ , 和blend weights,$W$ .输出是新的vertices.原始的定点坐标$\overline{t_i}$ 被转换为$\overline{t_i^,}$   

$\begin{aligned}
\overline{\mathbf{t}}_{i}^{\prime} &=\sum_{k=1}^{K} w_{k, i} G_{k}^{\prime}(\vec{\theta}, \mathbf{J}) \overline{\mathbf{t}}_{i} \\
G_{k}^{\prime}(\vec{\theta}, \mathbf{J}) &=G_{k}(\vec{\theta}, \mathbf{J}) G_{k}\left(\vec{\theta}^{*}, \mathbf{J}\right)^{-1} \\
G_{k}(\vec{\theta}, \mathbf{J}) &=\prod_{j \in A(k)}\left[\begin{array}{c|c}
\exp \left(\vec{\omega}_{j}\right) & \mathbf{j}_{j} \\
\hline \overrightarrow{0} & 1
\end{array}\right]
\end{aligned}$ 

$G_{k}^{\prime}(\vec{\theta}, \mathbf{J})$ 就是根据pose $\vec{\theta}$ 和原始链接点$J$ 转换生成新的连接点位置。

$A(k)$ 表示连接点k的有序父节点集合。

为了保证兼容性，我们使用最基本的蒙皮方法，并且学习一个function用来预测连接点的坐标，那最终model的形式，如下：

$\begin{aligned}
M(\vec{\beta}, \vec{\theta}) &=W\left(T_{P}(\vec{\beta}, \vec{\theta}), J(\vec{\beta}), \vec{\theta}, \mathcal{W}\right) \\
T_{P}(\vec{\beta}, \vec{\theta}) &=\overline{\mathbf{T}}+B_{S}(\vec{\beta})+B_{P}(\vec{\theta})
\end{aligned}$

其中$B_S(\vec{\beta})$ 和 $B_P(\vec{\theta})$ 分别表示形状参数和姿态参数对原始模板的修正量。

我们把他们分别叫做shape和pose blend shape.

所以最后的顶点坐标$\overline{t_i}$的表达式为：

$\overline{\mathbf{t}}_{i}^{\prime}=\sum_{k=1}^{K} w_{k, i} G_{k}^{\prime}(\vec{\theta}, J(\vec{\beta}))\left(\overline{\mathbf{t}}_{i}+\mathbf{b}_{S, i}(\vec{\beta})+\mathbf{b}_{P, i}(\vec{\theta})\right)$

这样最后的blend skinning 就融合了pose 和 shape。

下面继续详细分析：

#### Shape blend shapes：

不同人的体型（body shape）可以用一个线性函数$B_s$ 表示为：

$B_{S}(\vec{\beta} ; \mathcal{S})=\sum_{n=1}^{|\vec\beta|} \beta_{n} S_{n}$

其中：

$\vec\beta =[\beta_1,...,\beta_{|\vec\beta|}]$ ,code中它的长度是10；

$S_n\in \mathbb{R}^{3N}$ 表示shape displacement的正交主成分。

$S = [S_1,...,S_{|\vec{\beta}|}]\in\mathbb{R}^{3N*|\vec\beta|}$ ,$S$是通过配准的mesh训练得到的。

#### Pose blend shapes：

定义一个非线性函数，$R(\vec\theta)$ ，就是上面提到的罗德里格斯公式。将3维的轴角转化为9维的rotation matrices.

和之前工作不同的是，我们将pose blend shapes的影响定义为线性的，

$R^*(\vec\theta) = (R(\vec\theta)-R(\vec\theta^*))$

$\theta^*$ 表示静息姿态。

那么vertex相对于静息状态的偏移量为：

$B_{P}(\vec{\theta} ; \mathcal{P})=\sum_{n=1}^{9 K}\left(R_{n}(\vec{\theta})-R_{n}\left(\vec{\theta}^{*}\right)\right) \mathbf{P}_{n}$

$P_n\in \R^{3N}$ 还是shape displacement，是需要学习得到的。

#### Joint locations：

不同的body shape的关节位置不同，每个关节在rest pose中是一个3D的位置。这里，将关节定义为body shape$\vec\beta$ 的函数。

$J(\vec{\beta} ; \mathcal{J}, \overline{\mathbf{T}}, \mathcal{S})=\mathcal{J}\left(\overline{\mathbf{T}}+B_{S}(\vec{\beta} ; \mathcal{S})\right)$ 

$\mathcal{J}$是从rest vertices到rest joints的变换矩阵，$\mathcal{J}$是从很多来自不同人的pose中学习得到的；

#### SMPL model：

最终SMPL模型的$\Phi$ 的所有参数为：

$\Phi=\{\bar{T}, \mathcal{W}, \mathcal{S}, \mathcal{J}, \mathcal{P}\}$

$\overline{T}$ 是顶点集合；

$\mathcal{W}$ 是blend weight;

$\mathcal{S}$ 是shape displacement 矩阵，由体型差异造成；

$\mathcal{J}$ 是rest pose下顶点到关节点的变换矩阵；

$\mathcal{P}$是和$\mathcal{S}$ 类似的shape displacement,由姿态差异造成。

这些参数通过训练和学习得到，一旦训练完成后这些参数就固定下来。后面通过变化

$\vec{\beta}$ 和 $\vec{\theta}$ 来创建特定体型的任务模型和驱动动画。

最终SMPL模型的定义：

$M(\vec{\beta}, \vec{\theta}, \Phi)=W\left(T_{P}(\vec{\beta}, \vec{\theta} ; \bar{T}, \mathcal{S}, \mathcal{P}), J(\vec{\beta} ; \mathcal{J}, \bar{T}, \mathcal{S}), \vec{\beta}, \mathcal{W}\right)$

那么对于mesh中的一个顶点，序号为i,所做的变换为：

$t_{i}^{\prime}=\sum_{k=1}^{K} w_{k, i} G_{k}^{\prime}(\vec{\theta}, J(\vec{\beta} ; \mathcal{J}, \bar{T}, \mathcal{S})) t_{P, i}(\vec{\beta}, \vec{\theta} ; \bar{T}, \mathcal{S}, \mathcal{P})$ 

其中$w_{k,i}$ 是blend weight,$G_k^{\prime}$ 是从父关节点到当前关节点“累计”旋转变换并除去初始变换的一个“变换偏移量”，$t_{p,i}$ 是顶点初始状态+体型shape差异变形+姿态pose差异变形：

$t_{P, i}(\vec{\beta}, \vec{\theta} ; \bar{T}, \mathcal{S}, \mathcal{P})=\bar{t}_{i}+\sum_{m=1}^{|\beta|} \vec{\beta} s_{m, i}+\sum_{n=1}^{9 K}\left(R_{n}(\vec{\theta})-R_{n}\left(\overrightarrow{\theta^{*}}\right)\right) p_{n, i}$ 

#### 训练

SMPL参数的训练过程是在shape和pose数据集上最小化重建误差得到的。

- multi-pose数据集用来训练$\mathcal{S}$,包含40个人的1786个registration(registration是指对齐好的mesh)。
- multi-shape数据集用来训练$\mathcal{P}$,数据来自数据集CAESA,包含1700个男性registration和2100个女性registration。

现在分别使用$V_j^P$ 和 $V_j^S$ 表示multi-pose和 multi-shape数据集中的第j个mesh。

- 需要优化的参数是：$\Phi=\{\overline{T},\mathcal{W},\mathcal{S},\mathcal{J},\mathcal{P}\}$

- 优化目标是最小化顶点重建误差；

  文章中首先使用multi-pose数据集优化$\{\mathcal{J},\mathcal{W},\mathcal{P}\}$,然后使用multi-shape数据集优化$\{\overline{T},\mathcal{S}\}$ 。

  男性和女性的模型分别优化，分别得到$\Phi_m$ 和 $\Phi_f$ 。

(1) Pose Parameter Training

pose parameter主要是训练$\{\{{\mathcal{J},\mathcal{W},\mathcal{P}}\}$(joint location predict, blend weight和pose displacement)。为了达到这个目的，我们需要计算每个rest templates,$\hat{T}_i^P$ 和连接点位置，$\hat{J}_i^P$ 还有每个registration的的pose parameters $\vec{\theta_j}$. 前面已经说过multi-pose数据集包含了40个人的1786个registration，这里用下标$i$ 表示第$i$个人，下标$j$表示第$j$个registration。在pose数据集中，不同的registration的姿态是不同的，表示为$\vec{\theta}_j$ .

根据前面说的$W(.):$

$\begin{aligned}
M(\vec{\beta}, \vec{\theta}) &=W\left(T_{P}(\vec{\beta}, \vec{\theta}), J(\vec{\beta}), \vec{\theta}, \mathcal{W}\right)\end{aligned} $

这里可以写出在pose数据集上$W(.)$的形式为：

$\left.W\left(\hat{T}_{s(j)}^{P}+B_{P}(\vec{\theta} ; \mathcal{P}), \vec{\theta}\right), \hat{J}_{s(j)}^{P}, \vec{\theta}, \mathcal{W}\right)$

其中：

- $\hat{T}_{s(j)}^P$ 表示pose数据集中第j个mesh所对应的人物的rest template（是人工设计的）;

- $\hat{J}_{s(j)}^P$ 表示pose数据集中第j个mesh所对应的人物的joint location（同上）.

- $B_P(\vec{\theta};\mathcal{P})$ 前面说过，是在$\vec{\theta},\mathcal{P}$的作用下产生的顶点形变；

  综合上面所说的$W(.)$ 的过程就是当前（第j个）mesh 所属人物在姿态$\vec{\theta}$ 作用下产生的变形，而这个变形的过程涉及到的一些参数是需要学习的，这里就是$\mathcal{P}$ 和 $\mathcal{W}$ 。变形之后得到一个新的mesh,学习的目标就是找到最佳的参数使变形之后的的结果mesh和数据集中对齐好的mesh之间的差异最小。

  为了完成这个学习过程，定义一个包含数据项和正则项的目标函数：

  $E = E_D + \lambda_YE_Y + \lambda_JE_J + \lambda_PE_P + E_W$

  其中：

  - $E_D$ 是数据项，就是上面说的经过变形后的mesh与已经对齐的mesh顶点之间的差异；
  - $E_Y$是对称正则项；
  - $E_J$ 是joint正则项；
  - $E_P$ 是$\mathcal{P}$ 正则项；
  - $E_W$ 是$\mathcal{W}$ 正则项；

  下面对目标函数中的这几项分别进行阐述：

  （1）$E_D$

  $\left.E_{D}\left(\hat{T}^{P}, \hat{J}^{P}, \mathcal{W}, \mathcal{P}, \Theta\right)=\sum_{j=1}^{P_{r e g}} \| V_{j}^{P}-W\left(\hat{T}_{s(j)}^{P}+B_{P}(\vec{\theta} ; \mathcal{P}), \vec{\theta}\right), \hat{J}_{s(j)}^{P}, \vec{\theta}, \mathcal{W}\right) \|^{2}$

  其中：

  $\Theta = {\vec{\theta_1},...,\vec{\theta_{P_{reg}}}}$ ,$P_reg$ 是pose训练集中registration的数目；

  $P_{subj}$ 是训练集中人物的数目；

  $\hat{T}^P=\{\hat{T}_i^P\}_{i=1}^{P_{subj}}$ ,是训练集中mesh template的集合。

  $\hat{J}^P = \{\hat{J}_i^P\}_{i=1}^{P_{subj}}$ 是mesh template的joint的集合。

  （2）$E_Y$

  $E_Y$是对称正则项，也就是激励template mesh和joint为对称的，对左右不对称的情况进行惩罚：

  $E_{Y}\left(\hat{J}^{P}, \hat{T}^{P}\right)=\sum_{i=1}^{P_{\text {subj }}} \lambda_{U}|| \hat{J}_{i}^{P}-U\left(\hat{J}_{i}^{P}\right)\left\|^{2}+\right\| \hat{T}_{i}^{P}-U\left(\hat{T}_{i}^{P}\right) \|^{2}$

  其中 U ( T ) U(T)U(T)的作用是沿着矢状面(sagittal plane 指将躯体纵断为左右两部分的解剖平面)左右翻转人体；

  (3)$E_J$

  文中将人体分割为24个部分，根据这个分割情况可以得到不同部分之间的连接的一个“顶点序列”，是一个环状的顶点集合，使用一个回归器$\mathcal{J}_I$  计算joint的初始中心，实际上就是对这个顶点集合里面的顶点的坐标做平均。当估计joint中心的时候，锁甲的正则项就是让估计的中心接近这个初始中心：
  $E_{J}\left(\hat{T}^{P}, \hat{J}^{P}\right)=\sum_{i=1}^{P_{s w b j}}\left\|\mathcal{J}_{I} \hat{T}_{i}^{P}-\hat{J}_{i}^{P}\right\|^{2}$

  (4)$E_P$

  为了防止pose-dependent blend shape的过拟合，这里对$\mathcal{P}$也做了一个正则化，使其趋向于0：

  为了防止pose-dependent blend shape的过拟合，这里对P \mathcal{P}P也做了一个正则化，使其趋向于0：

  $E_P(P)=||\mathcal{P}||_F^2$


  其中$||·||_F$是Frobenius范数，实际上就是矩阵中对应元素的平方和再开平方.

  (5)$E_W$

  根据人体的分割结果，通过"diffus"可以得到一个初始的blend weight于$\mathcal{W}_I$
   ,现在添加一个正则项使$\mathcal{W}$趋向于$\mathcal{W}_I$
  $E_W(\mathcal{W})=||\mathcal{W}-\mathcal{W}_I||_F^2$
  	

  #### Joint Regressor

  通过上面的优化过程可以得到训练集中每个人物的template mesh和joint location。但是如果我们想为新的人物预测其关节位置呢？文章中通过学习一个regressor matrix $ \mathcal{J}$预测Joint位置。$\mathcal{J}$通过非负最小二乘计算得到，并使权重加起来为1。这种方法使得计算joint的顶点是稀疏的，同时权重非负和加起来为1又使得预测的joint不会出现在mesh的外侧。

  (2)Shape parameter training

  shape 空间是通过"mean and principal shape direction" $\{\overline{T},\mathcal{S}\}$来定义的。计算方法multi-pose数据集中shape做一个pose归一化，然后再运行PCA得到。
  pose归一化的过程是将数据集中的registration $ V_j^S$转换为一个处于rest pose $\vec{\theta^*} $下的registration $ \hat{T}_j^S $;转为rest pose这一步保证了pose和shape的建模不会相互影响。那么如何进行pose归一化呢？对于数据集中的一个一个registration  $V_j^S$,首先要估计它的姿势，也就是要寻找一个姿势表示$\vec{\theta}$
   使得经过这个参数变换后的mesh和原始的mesh的误差最小，也就是优化：
  $\vec{\theta}_j = \mathop{\arg\min}_{\vec{\theta}} \sum_{e}||W_e(\hat{T}_{\mu}^P+B_P(\vec{\theta};\mathcal{P})),\hat{J}_{mu}^P,\vec{\theta},\mathcal{W} - V_{j,e}^S||^2$


  其中：

   $\hat{T}_{\mu}^P $是multi-pose数据集中的mean pose;
  $P \hat{J}_{\mu}^P $是multi-pose数据集中的mean joint location;
  $V_{j,e}^S \in \mathbb{R}^3$“an edge of the registration”，是通过一对相邻顶点之间相减得到的，就是当前shape和mean shape的对应顶点坐标相减。通过对mesh中的所有的edge求和，可以在不知道人物具体的shape的情况下求得一个姿态的较好估计。
  得到姿态 $\vec{T}_j^S$	之后，可以求 $\hat{T}_j^S$

  $\hat{T}_j^S = \mathop{\arg\min}_{\vec{T}}||W(\hat{T}+B_p(\vec{\theta}_j;\mathcal{P}),\mathcal{J}\hat{T},\vec{\theta},\mathcal{W}) - V_j^S||^2$


  然后在$\{\hat{T}_j^S\}_{j=1}^{S_{subj}}$ 上运行PCA,得到 $\{\overline{T},\mathcal{S}\}$PCA这一步是为了最大大rest pose下顶点偏移的可解释方差(explained variance)，同时使shape direction的数目较少。




#### Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image(SMPLify)

### Abstract:

本文使用一张2d人体图像拟合出人体的3d形状和3d姿态。先使用DeepCut检测出2d人体关键点，然后再将3d人体模型SMPL和2d关键点拟合。拟合是通过最小化2d关键点和3d关键点间的误差实现的。

### Introduction:

使用3d人体模型的好处：

1.能够获得身体形状信息

2.可以reason aboout interpenetration问题,即2d映射到3d的过程中会出现一些不可能的姿势。之前的一些工作使用3d棍图（stick figures）就可能出现这些问题。可以通过一个胶囊模型（“capsules”）来解决interpenetration的问题。

为了解决pose的二义性，还需要pose prior.



method:

![image-20210723151425435](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210723151425435.png)

Fig.2展示了系统的整体流程，输入一张2d图片，使用DeepCut 预测出2D关键点，$J_{est}$ . 每一个关键点$i$ 会提供一个置信度$w_i$.然后和3d模型的3d关键点进行拟合，拟合出3d模型。



$J(\beta)$ :利用shap参数$\beta$ ，预测shape,然后再通过shape, 预测3D 关键点.

通过一个全局刚性变换，可以控制关节摆出任意姿势。

$R_\theta(J(\beta)_i)$ :代表第$i$个3d关节点的坐标，表示加上pose之后的关键点位置。。$R_\theta$ 代表由pose参数$\theta$ 推导出来的刚性变换。

我们将DeepCut生成的关键点和SMPL的关节点关联起来，为了将3D投影到2D，我们使用了一个投影相机模型，使用参数K控制。

3.1使用胶囊近似身体

我们训练了一个回归模型，从shap参数回归出胶囊的（轴长和半径），并通过$R_\theta$ 控制胶囊模型的姿势。

首先将胶囊和模板的身体关节点绑定，使用基于梯度的优化方法优化（radii and axis lengths）使身体和胶囊之间的双向距离最小。我们使用交叉验证岭回归学习到一个从shap 参数$\beta$ 到 radii and axis lengths的线性回归。 

3.2 目标函数

为了将3d pose和shape拟合到2d关键点，我们需要最小化一个包含5个错误项之和的目标函数：

$E(\beta,\theta)=E_{J}\left(\boldsymbol{\beta}, \boldsymbol{\theta} ; K, J_{\text {est }}\right)+\lambda_{\theta} E_{\theta}(\boldsymbol{\theta})+\lambda_{a} E_{a}(\boldsymbol{\theta})+\lambda_{s p} E_{s p}(\boldsymbol{\theta} ; \boldsymbol{\beta})+\lambda_{\beta} E_{\beta}(\boldsymbol{\beta})$

一个关节点项，三个姿态先验,一个形状先验.

$E_{J}\left(\boldsymbol{\beta}, \boldsymbol{\theta} ; K, J_{\text {est }}\right)=\sum_{\text {joint } i} w_{i} \rho\left(\Pi_{K}\left(R_{\theta}\left(J(\boldsymbol{\beta})_{i}\right)\right)-J_{\text {est }, i}\right)$ 

关节点项惩罚2d关键点和投影关键点之间的距离。其中$\Pi_{K}$ 表示利用相机参数K,将3D关键点投影到2D关键点，$w_i$ 为上文提到的关键点的置信度。对于遮蔽的关键点，置信度往往是比较低的，在这种情况下pose是由 pose prior实现的。同时为了印制噪音，使用了鲁棒代价函数Geman-McClure ，$\rho$ .

$E_{a}(\boldsymbol{\theta})=\sum_{i} \exp \left(\boldsymbol{\theta}_{i}\right)$ 

 惩罚手肘和膝盖的不正常弯曲。因为在关节不弯曲的情况下，角度$\theta_i$ 是零，正常弯曲时角度为负，不正常弯曲时角度为正。

pose prior:

和之前的许多工作一样，使用CMU 数据集来训练pose prior.为了构造先验，我们使用MoSh将CMU marker data 拟合到SMPL,获得poses.然后使用一个混合高斯模型，去拟合100个subjects的100万姿势。

$\begin{aligned}
E_{\theta}(\boldsymbol{\theta}) \equiv-\log \sum_{j}\left(g_{j} \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\mu}_{\theta, j}, \Sigma_{\theta, j}\right)\right) & \approx-\log \left(\max _{j}\left(c g_{j} \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\mu}_{\theta, j}, \Sigma_{\theta, j}\right)\right)\right) \\
&=\min _{j}\left(-\log \left(c g_{j} \mathcal{N}\left(\boldsymbol{\theta} ; \boldsymbol{\mu}_{\theta, j}, \Sigma_{\theta, j}\right)\right)\right)
\end{aligned}$

其中，$g_j$ 是混合N=8的高斯模型的权重

capsule approximation:

$E_{s p}(\boldsymbol{\theta} ; \boldsymbol{\beta})=\sum_{i} \sum_{j \in I(i)} \exp \left(\frac{\left\|C_{i}(\boldsymbol{\theta}, \boldsymbol{\beta})-C_{j}(\boldsymbol{\theta}, \boldsymbol{\beta})\right\|^{2}}{\sigma_{i}^{2}(\boldsymbol{\beta})+\sigma_{j}^{2}(\boldsymbol{\beta})}\right)$

shape prior:

$E_{\beta}(\boldsymbol{\beta})=\boldsymbol{\beta}^{T} \Sigma_{\beta}^{-1} \boldsymbol{\beta}$

 

#### Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop（SPIN）





