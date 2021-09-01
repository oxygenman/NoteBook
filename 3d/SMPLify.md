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

关节点项惩罚2d关键点和投影在SMPL关键点之间的距离。其中$\Pi_{K}$ 表示利用相机参数K,将3D关键点投影到2D关键点，$w_i$ 为上文提到的关键点的置信度。对于遮蔽的关键点，置信度往往是比较低的，在这种情况下pose是由 pose prior实现的。同时为了印制噪音，使用了鲁棒代价函数Geman-McClure ，$\rho$ .

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

我们定义了一个惩罚互相贯穿的错误项（interpenetration），我们将错误项和胶囊之间的相交体积关联起来。因为胶囊之间的交集不好计算，所以我么使用球形来近似胶囊，球形的球心$C(\theta,\beta)$ ,球形的半径$r(\beta)$ 对应胶囊的半径。We consider a 3D isotropic Gaussian with $\sigma(\beta)=\frac{\gamma(\beta)}{3}$ for each sphere, and define the penalty as a scaled version of the integral of the product of Gaussians corresponding to “incompatible” parts。

$E_{s p}(\boldsymbol{\theta} ; \boldsymbol{\beta})=\sum_{i} \sum_{j \in I(i)} \exp \left(\frac{\left\|C_{i}(\boldsymbol{\theta}, \boldsymbol{\beta})-C_{j}(\boldsymbol{\theta}, \boldsymbol{\beta})\right\|^{2}}{\sigma_{i}^{2}(\boldsymbol{\beta})+\sigma_{j}^{2}(\boldsymbol{\beta})}\right)$

shape prior:

(看不懂)

$E_{\beta}(\boldsymbol{\beta})=\boldsymbol{\beta}^{T} \Sigma_{\beta}^{-1} \boldsymbol{\beta}$

