#                                                    mtcnn详解

------

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-8e2d0268a6105c44af8512fbf24a3321_720w.jpg" alt="img" style="zoom: 80%;" />

## 主要工作

1. 采用三级级联卷积神经网络架构
2. 多任务学习:联合人脸检测框和人脸关键点进行学习
3. 提出一种在线难例挖掘策略  

### 工作成果

- **人脸检测**

| ![image-20201110140821306](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201110140821306.png) | ![image-20201110140925629](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201110140925629.png) | ![image-20201110141111114](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201110141111114.png) |
| ------------------------------------------------------------ | :----------------------------------------------------------: | ------------------------------------------------------------ |
|                                                              |               **在widerface数据集上的pr曲线**                |                                                              |

- **人脸对齐**

  ![image-20201111201401030](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201111201401030.png)

  评价标准:the inter-ocular distance normalized error
  $$
  e_{i}=\frac{\left\|\mathrm{x}_{i}-\mathrm{x}_{i}^{*}\right\|_{2}}{d_{I O D}}
  $$
  选取n张人脸求分子为两点之间的距离,分母为两眼中心的间距,目的是对距离做一个归一化屏蔽脸部大小问题.

- **算法效率**

  在2.60GHz CPU上16fps, 在GPU (Nvidia Titan Black)上99fps.(matlab 代码).

  ### 总体流程

  

### 三级级联网络架构

![image-20201111191727156](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201111191727156.png)

### 第一级网络P-Net:

在原论文中P-Net是一个三层





## 参考:

[论文链接:Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878.pdf) 

