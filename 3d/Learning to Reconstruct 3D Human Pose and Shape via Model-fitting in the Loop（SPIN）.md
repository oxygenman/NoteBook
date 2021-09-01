## Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop（SPIN）

[参考](https://blog.csdn.net/JerryZhang__/article/details/109535563)

### 主要工作：

本论文主要是在<<End-to-end recovery of human shape and pose>>[1]的基础上，又增加了一个监督信号，即将基于优化的SMPLify方法生成的SMPL模型参数,用于监督[1]的回归模型，用[1]回归生成的模型参数来初始化SMPLify的模型，两者循环迭代，相互促进，相互提高。

![image-20210901150756648](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210901150756648.png)

### 主要方法：

#### Regression network：

回归网络部分采用和[1]相同的网路结构， 不同的是本文采用了Zhou et al [参考]提出的3D 旋转表示方式来替代原有的方式。回国模型的输出是：

$\Theta_{reg} = \{\theta_{reg},\beta_{reg}\}$  和 相机参数 $\Pi_{reg}$ .

利用这些参数我们可以投影出2d关键点$J_{reg} = \Pi_{reg}(X_{reg})$ 

利用SMPL模型可以生成3D Mesh. $M_{reg} = \mathcal{M}(\theta_{reg},\beta_{reg})$

利用投影出的2D的关键点可以计算重投影损失，但是本文这种损失加重了网络的负担，迫使网络去寻找和2D 关键点ground thruth相匹配的3d姿态。

#### Optimization routine:

优化方式部分基本继承SMPLify的工作，SMPLify的目标函数由关键点重投影损失和几个pose和shape的先验组成。

$E_{J}\left(\beta, \theta ; K, J_{e s t}\right)+\lambda_{\theta} E_{\theta}(\theta)+\lambda_{a} E_{a}(\theta)+\lambda_{\beta} E_{\beta}(\beta)$ 

$\theta$和$\beta$分别是SMPL模型的pose和shape参数；
$K$是摄像机参数；
$ E_{\theta}(\theta)$是shape拟合的混合高斯先验；

$E_{\alpha}(\theta)$ 是手肘和膝盖关节的不正常旋转的惩罚；
$E_{\beta}(\beta)$ 是shape系数的二次惩罚项；

本论文中作何没有使用SMPLify中的 interpenetration error项，作者说它使拟合过程变慢，并且对结果提升不多。作者还对SMPLify的训练过程进行了改进，将单张图片的推导改成batch推导，并在GPU上进行训练。作者还改进了2d关键点的使用方式。

#### SPIN:

SPIN就是上面两个过程的结合，过程描述如下：

一张图片经过回归网络生成$\Theta_{reg}$ ,使用$\Theta_{reg}$ 初始化optimization routine使用的模型，然后optimization部分又迭代优化出$\Theta_{opt} = \{\theta_{opt},\beta_{opt}\}$ ,并生成$M_{reg} = \mathcal{M}(\theta_{opt},\beta_{opt})$ .

有了优化后的模型参数又可以优化回归网络：

$L_{3D} = ||\Theta_{reg} - \Theta_{opt}||$

 $L_{3D} = ||M_{reg} - M_{opt}||$


