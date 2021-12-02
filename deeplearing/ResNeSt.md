## 2ResNeXt

https://zhuanlan.zhihu.com/p/32913695

ResNext的基本结构：

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-ee942d228efdaaff277c8a9a8b96a131_720w.jpg)

上右图就是rexnext的基本结构，旁边的residual connection 就是公式中的x直接连过来，然后剩下的是32组独立的同样结构的变换，最后再将得到的特征直接相加，符合split-transform-merge的模型。

ResNext的优化实现方式：

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-324282edcbfbd476fbe406c5fddb8dfd_720w.jpg)

a是ResNext基本单元，如果把输出那里的1x1合并到一起，得到等价网络b拥有和Inception-ResNet相似的结构，得到等价网络c则和通道分组卷积的网络有相似的结构。到这里，可以看到本文的野心很大，相当于在说，Inception-ResNet和通道分组卷积网络，都只是RexNext这一范式的特殊形式而已，进一步说明了split-transform-merge的普遍性和有效性，以及抽象程度更高，更本质一点儿。

resnext block的pytorch 实现：

在实现block的时候，需要传入的控制参数有输入维度，cardinality,transform滤波器的数量。

```python
class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls*2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls*2)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls, self.group_chnls*2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_chnls*2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return F.relu(out)
```

## ResNeSt

https://www.cnblogs.com/xiximayou/p/12728644.html

ResNeSt是基于SENet和SKNet ResNeXt发展而来,我们先分别看下这几个网络；

### [SENet](https://zhuanlan.zhihu.com/p/32702350) 

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-eb33a772a6029e5c8011a5ab77ea2f74_720w.jpg)

上图是SENet的主要组成部分SEBlock,其中$F_{tr}$ 是传统的卷积结构。途中的$F_{sq}(.)$ 被作者称为是Squeeze的过程，其实就是对U先做一个Global Average Pooling,输出的1x1xC数据再经过两级全连接（途中的Fex(.),作者称为Excitation过程），最后使用sigmoid将输出限制到[0,1]的范围，把这个值作为scale乘到U的C个通道上，作为下一级的输入数据。这种结构的想法很简单，就是希望通过学到的scale的值来把重要通道的值增强，不重要的通道减弱。

还有一个需要注意的地方就是Excitation部分使用两个全连接来实现，第一个全连接把C个通道压缩成C/r个通道来降低计算量（后面跟RELU）,第二个全连接再恢复回C个通道（后面跟sigmoid值），r是指压缩的比例，作者尝试了r在各种取值下的性能，最后得出结论r=16时整体性能和计算量最平衡。

将SEBlock 插入到inception和resnet中

![这里写图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20170916211417675)

![这里写图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20170916211433498)

SEBlock的实现代码:

```python
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filter3,filter3//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3//16,filter3,kernel_size=1),
            nn.Sigmoid()
        )
```

[SENet](https://blog.csdn.net/luxinfeng666/article/details/102070894) 

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20191004161740356.png)

SKNet模块如上图所示。Split操作是将原feature map 分别通过一个3x3的卷积和3x3的空洞卷积（感受野为5x5）生成两个feature map: $\hat{U}$ 和 $\widetilde{U}$ 。然后将这两个feature map进行相加。生成U。生成的U通过Fgp函数（全局平均池化）生成1x1xC的feature map,(图中s),该feature map 通过Ffc函数（全连接层）生成dx1的向量（图中z）,公式如图中所示（$\delta$ 表示relu激活函数，B表示Batch Normalization, W是一个dxC维的）。d的取值是由公式d=max（C/r,L）确定，r是一个缩小的比率，L表示D的最小值。生成的z通过$a_c$ 和 $b_c$ 两个函数，并将生成的函数值与原先的$U_1$ 和 $U_2$ 相乘，由于$a_c$ 和 $b_c$ 的函数值相加等于1，因此能够实现对分支中的feature map设置权重，因为不同的分支卷积核尺寸不同，因此实现了让网络自己选择合适的卷积核。

```
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V

```

### ResNest

[知乎](https://zhuanlan.zhihu.com/p/133805433)

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/1503039-20200418213847824-757645072.png)

首先是借鉴了ResNeXt网络的思想，将输入分为K个，每一个记为Cardinal1-k,然后又将每个Cardinal拆分为R个，每一个记为Split1-r,所以总共有G=KR个组。

然后每一个Cardinal中的split attention具体结构如下：

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/1503039-20200418213924932-1549629534.png)



#### 更高效的Radix-major Implementation

![image-20210909145641126](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210909145641126.png)

我们称图（fig1 右）所示的结构为cardinality-major实现。这种实现比较直观，但是但是他无法使用标准的CNN操作进行模块化和加速。因此我们引入了一个radix-major实现。如figure 4所示。

首先输入特征图被分为RK groups. 每一个group都有一个cardinality index 和一个 radix index.在这种结构中，具有相同radix index的group相邻。接着，我们可以对不同的splits执行一个加和操作，这样的化，具有不同radix index但是拥有相同cardianlity index的groups被融合在了一起。接着执行一个全局池化操作，等同于分别对每一个cardianl group执行全局池化操作，再拼接在一起。接着两层group数等同于cardinality的全连接操作，用来预测每个splits 的 attention weights.。将全连接层分组使得每一个cardinal group的attention weights是独一无二的。

另外，在这种实现中，第一层的1x1卷积可以被统一到一层中，3X3可以使用一层分组卷积来实现，分组个数为Rk。





#### split  attention block

```python
class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()
```

[商品识别挑战赛](https://cloud.tencent.com/developer/article/1779175)