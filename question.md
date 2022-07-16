1.目标检测中数据增强的方法有哪些？优缺点？

[数据增强包](https://www.freesion.com/article/13111107879/#CoarseDropout%C2%A0%E5%9C%A8%E5%9B%BE%E5%83%8F%E4%B8%8A%E7%94%9F%E6%88%90%E7%9F%A9%E5%BD%A2%E5%8C%BA%E5%9F%9F)

[CV中有哪些数据增强方法](https://www.zhihu.com/question/319291048/answer/2258940108)

cutout 就是将图像切去一块儿 可以用来增强数据遮挡的情况

mixup 即将两张图片的像素按一定比例相加融合，你中有我我中有你。可以增强模型的泛化能力，并且能够提高模型对于对抗攻击的鲁棒性 [关于mixup方法的一个综述](https://zhuanlan.zhihu.com/p/439205252) 

cutmix 结合了cutout和mixup 将原图切块，用另一张图的一部分补上

Mosaic 是将四张图片合成一张图片，可以变相增大batchsize

2.激活函数有哪些？优缺点

[激活函数总结（持续更新）](https://zhuanlan.zhihu.com/p/73214810)

[在使用relu的网络中，是否还会存在梯度消失的问题](https://www.zhihu.com/question/49230360/answer/114914080)

3.分类的损失函数？回归的损失函数？
4.梯度下降？几种优化器?

[NAG详解](https://maimai.cn/article/detail?fid=1611261762&efid=QG9uqqsOrTdPa8hTbppqIg).

[机器学习优化器总结](https://zhuanlan.zhihu.com/p/150113660)

NAG(Nesterov Accelerated Gradient) NAG的本质上是多考虑了目标函数的二阶导信息

NAG的原始形式：

$\begin{aligned}
d_{i} &=\beta d_{i-1}+g\left(\theta_{i-1}-\alpha \beta d_{i-1}\right) \\
\theta_{i} &=\theta_{i-1}-\alpha d_{i}
\end{aligned}$ 

可以变化为：

$\begin{array}{l}
d_{i}=\beta d_{i-1}+g\left(\theta_{i-1}\right)+\beta\left[g\left(\theta_{i-1}\right)-g\left(\theta_{i-2}\right)\right] \\
\theta_{i}=\theta_{i-1}-\alpha d_{i}
\end{array}$ 

相比于原始的momentum 多了后面一项，直观的理解的话，除了考虑动量和梯度，还要考虑上两次梯度的差值。如果插值和动量方向一直就再进一步，如果不一致则减缓动量。

Adagrad 是一种自适应学习率的方法，通过引入梯度平方的累加和，来缩放学习率。对于梯度更新较多的降低较多学习率，对于梯度更新较小的，降低较小学习率，直到学习率降为0.

所以Adagrad的缺点在于随着迭代次数增多，学习率会越来越小。

RMSprop采用指数衰减平均的方式取代Adgrad取代所有梯度历史平方值的总和的平方根。

Adam 结合了momentum和RMSprop,同时对梯度进行一阶矩估计和二阶矩估计。

5.几种归一化方法？

[归一化 标准化 白化](https://zhuanlan.zhihu.com/p/475106090)

6.各种评价指标？

[机器学习 F1-Score, recall, precision_Matrix_11的博客-CSDN博客_fi score](https://blog.csdn.net/matrix_space/article/details/50384518)

[目标检测评价指标](https://zhuanlan.zhihu.com/p/88896868)

7.如何解决小目标检测的问题？

https://zhuanlan.zhihu.com/p/121666693

https://zhuanlan.zhihu.com/p/83220498

８．ｍtcnn和retinaface的优缺点？

9.混淆矩阵

