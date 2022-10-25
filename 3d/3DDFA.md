Face Alignment in Full Pose Range: A 3D Total Solution

3D Dense Face Alignment (3DDFA)

3.1 3D morphable Model 

3.1.1 Rotation Formulation   四元数 欧拉角

3.2 Feature Design

1. model view feature: Pose Adaptive Feature(PAF)

   ![image-20220727170909014](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220727170909014.png)

   第一步，基于当前的参数p,使用3DMM进行三维形状的构建，并计算每个点的柱坐标。在其上采样大小为64x64的feature anchor.

   第二步，3DMM模型投影到2D平面上，得到64x64x2的feature anchors,$V(p)_anchor$,如图fig.4(b)所示，分别表示可见与不可见的部分。

   第三步，在每个anchor裁剪dxd的patch，根据它们的圆柱体坐标，将这些patch拼接到成（64×d）x(64*d)的map，如图fig4(c)所示。

   第四步，在patch map 上执行dxd的卷积，得到64x64的response maps,如fig4(d)所示。

   作为后续卷积操作的特征输入。

   

2. image-view feature: Projected Normalized Coordinate Code (PNCC)

   ![image-20220728151849200](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220728151849200.png)

   将人脸mesh的顶点坐标归一化：

   $\mathrm{NCC}_{d}=\frac{\overline{\mathbf{S}}_{d}-\min \left(\overline{\mathbf{S}}_{d}\right)}{\max \left(\overline{\mathbf{S}}_{d}\right)-\min \left(\overline{\mathbf{S}}_{d}\right)} \quad(d=x, y, z),$

   我们叫做Normalized Coordinate Code(NCC).

   因为NCC和RGB一样，有三个通道，所以可以像rgb图片一样展示出来。

   

   在拟合的过程中，我们通过Z-Buffer去渲染使用NCC上色的3D投影人脸。

   $\begin{array}{c}
   \text { PNCC }=Z-B u f f e r\left(V_{3 d}(\mathbf{p}), \mathrm{NCC}\right) \\
   V_{3 d}(\mathbf{p})=\mathbf{R} *\left(\overline{\mathbf{S}}+\mathbf{A}_{i d} \boldsymbol{\alpha}_{i d}+\mathbf{A}_{e x p} \boldsymbol{\alpha}_{e x p}\right)+\left[\mathbf{t}_{2 d}, 0\right]^{\mathrm{T}}
   \end{array}$ 

   我们叫这个渲染的图片是Projected Normalized Coorinate Code.(PNCC)

3.3 网络设计

![网络架构图](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20200731161007624.png)

输入：当前参数p,以及人脸图像

上面的分支：基于参数P得到的PNCC的输入的人脸图像组合作为卷积神经网络的输入

下面的分支：基于参数P得到的带有feature anchor的图像输入PAC进行操作得到特征图PAF,将PAF作为后续卷积神经网络的输入。

结果：两个分支的结果通过最后的全连接层得到，$\Delta_p$,即$\Delta p = Net^k(PAF(p^k,I),PNCC(p^k,I))$ ,基于$\Delta p$ 更新参数，即$p^{k+1}=p^k+\Delta p^k$ 。 参数p主要包含以下信息：

- 6维的姿势参数$[q_0,q_1,q_2,q_3,t_{2dx},t_{2dy}]$ 
- 199维的identity参数$\alpha_{id}$ 
- 29维的expression参数$\alpha_{exp}$

3.4 代价函数设计

PDC(Parameter Distance Cost)

设计目标：使得目前参数与ground-truth之间的距离越小越好

$\mathrm{E}_{\mathrm{pdc}}=\left\|\Delta \mathrm{p}-\left(\mathrm{p}^{\mathrm{g}}-\mathrm{p}^{\mathrm{k}}\right)\right\|^{2}$ 

VDC 

设计目标：基于当前的参数投影到二维点，与ground-truth进行比较计算。

与前一个代价函数相比，有考虑到各个参数的语义。但是，这个函数本身不是凸函数，不能保证收敛。

$\mathrm{E}_{\mathrm{vdc}}=\left\|\mathrm{V}\left(\mathrm{p}^{0}+\Delta \mathrm{p}\right)-\mathrm{V}\left(\mathrm{p}^{\mathrm{g}}\right)\right\|^{2}$

WPDC

设计目标：希望对不同的参数赋予不同的权重，以表示这一参数的重要性。

![image-20220728165420580](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220728165420580.png)

在计算过程中，基于当前参数进行投影映射，将得到的二维图像与ground-truth进行比较。在计算过程中，只有第i个参数的值是基于前一轮参数值加上这一轮得到的$\Delta p$ ,其他参数均采用ground-truth的值。分母部分为w中的最大值。

虽然这一代价函数做了各个参数的重要性判断，但是忽略了参数之间的优先级问题。

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20200731163633176.png)

以上图为例。图a是一张张着嘴的接近只有侧脸的图像。使用这一代价函数时，会赋予表情参数以及旋转参数具有较高的重要性。图b是进行轮迭代后的结果。可以发现，在姿势足够准确之前就去估计表情参数没有太大的意义。图c是CNN只关注pose参数产生的结果。明显这一结果要优于图b.

那么基于此，对当前的代价函数进行优化，即OWPDC.

OWPDC

![image-20220728173636576](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20220728173636576.png)

这里的改进是，寻找是结果最优化的权重参数。