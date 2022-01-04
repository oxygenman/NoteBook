#  Retinaface详解

## 主要工作

（1）给WIDER FACE 数据集标注了人脸关键点

（2）添加了一个人自监督的人脸形状三维重建分支

（3）在WIDER FACE hard test set上ap达到91.4%

（4）提高了人脸verification的准确率 （TAR=89.59% for FAR=1e-6）

（5）使用了轻量backbone

(各种评价指标)https://zhuanlan.zhihu.com/p/87503403

## 实验结果

## 实现细节

#### 整体网络架构

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20201013213134781.png)

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20201013213635452.png)

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20201013213643304.png)

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20201013213654758.png)

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/20201013213702246.png)

#### 特征金字塔

论文中使用resnet backbone的输出构建了5层的FPN,我们一般使用的的pytorch代码使用mobilenet 作为backbone,构建了3层的FPN.

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-d136addbe3dc08f6cbae7233f753bf9a_720w.jpg)

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-fe85fb352b9c212fb6d5416330fad9d2_720w.jpg)

代码实现

```python
class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=(int(output2.size(2)), int(output2.size(3))), mode="nearest")
        #print("output2.size(2):",output2.size(2))
        #print("output2.size(3):",output2.size(3))
        #print("up3.size:",up3.size())
        #up3 = F.interpolate(output3, size=[40, 40], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=(int(output1.size(2)), int(output1.size(3))), mode="nearest")
        #print("output1.size():",output1.size(2))
        #print("output1.size(3):",output1.size(3))
        #up2 = F.interpolate(output2, size=[80, 80], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out
```



#### ContextModule

上下文模块来源于SSH和PyRamidBox.可以用来融合不同感受野，提高刚性上下文建模能力。并且retinaface还借鉴了WIDER  Face Challenge 2018冠军解决方案，将ContextModule中的3*3卷积替换成DCN(deformable convolution network).增强了非刚性建模能力。（我们在实际的pytorch代码中没有使用DCN）.

![image-20210824153654710](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20210824153654710.png)

我们实际使用的context module 如上图所示。

代码是先：

```python
class SSH(nn.Module):
    #这里都是3*3卷积不要被名字骗了
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out
```

#### DCN

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy84OTA0NzIwLTUyYTI3MjgwMjhhZTRiYWEucG5n)

可变形卷积就是先通过新加的一个卷积层来预测特征图上每个位置的偏移量，如上右图所示，偏移map的大小和原特征图一致，但是其维度为2维，因为涉及到x,y两个方向的偏移。偏移量往往不是整数，所以需要使用双线性插值来计算对应偏移位置的数值。双线性插值的计算过程如图所示。

![在这里插入图片描述](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy84OTA0NzIwLTJkZmU2NjIxMDAyYTUyZjUucG5n)

对于相邻的点来说x1-x0=1、y1-y0=1，所以可以继续简化成公式5

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy84OTA0NzIwLWUyNmFiYTRjNWQwNzlmZDUucG5n)

#### Dense Regression Branch

先重建出三维人脸，然后再将3维人脸根据相机参数投影到2d人脸上，并计算和原图的loss.具体过程因为复现代码中没有暂时略。

#### Multibox loss的实现

$$
\begin{aligned}
L &=L_{c l s}\left(p_{i}, p_{i}^{*}\right)+\lambda_{1} p_{i}^{*} L_{b o x}\left(t_{i}, t_{i}^{*}\right)+\lambda_{2} p_{i}^{*} L_{p t s}\left(l_{i}, l_{i}^{*}\right)+\lambda_{3} p_{i}^{*} L_{p i x e l}
\end{aligned}
$$

Retinaface中Multi-task Loss 分为四个部分:

(1)  $L_{c l s}\left(p_{i}, p_{i}^{*}\right):$  为softmax损失函数,即先算softmax,再算交叉熵.其中$p_i$为第i个priorbox是人脸的概率,$p_{i}^*$为groundtruth类别标签.

(2)$L_{b o x}\left(t_{i}, t_{i}^{*}\right):$  为smooth-L1 loss,用来回归检测框.

(3)$ L_{p t s}\left(l_{i}, l_{i}^{*}\right)$  :  为smooth-L1 loss,用来回归关键点儿.

(4)$L_{p i x e l}:$ 为3Dmask损失

```python
class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        #predictions:包含网络的三部分输出
        #假如batch szie是10,那么:
        #loc_data : [10,4200,4]
        #conf_data:  [10,4200,2]
        #lambda_data: [10,4200,10]
        loc_data, conf_data, landm_data = predictions
        #priors是之前根据各个featuremap生成的priorbox 数据
        priors = priors
        num = loc_data.size(0) # batch size
        num_priors = (priors.size(0)) # 每张图片产生的prior box的个数,这里是4200
        #num为batchsize()
        # match priors (default boxes) and ground truth boxes
        #[num,4200,4]
        #下面这三个是用来存储prior box 和 groundtruth 匹配之后生成的和网络输出结构一样的训练监督数据 
        loc_t = torch.Tensor(num, num_priors, 4)
        #[num,4200,10]
        landm_t = torch.Tensor(num, num_priors, 10)
        #[num,4200]
        conf_t = torch.LongTensor(num, num_priors)
        #遍历batch中的每一张图片,使用groundtruth和priorbox进行匹配
        for idx in range(num):
            #[boxnum,4]
            truths = targets[idx][:, :4].data
            #[boxnum,1]
            labels = targets[idx][:, -1].data
            #[boxnum,10]
            landms = targets[idx][:, 4:14].data
            #[4200,4]
            defaults = priors.data
            #匹配先验框和groundtruth找出正负样本
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        # mobilenet [1024,4200]
        pos1 = conf_t > zeros
        #batch中每张图片所包含的正样本个数
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        #batch 中所有正样本的个数
        N1 = max(num_pos_landm.data.sum().float(), 1)
        #numpriors->[numpriors,10]
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        #计算landmark的loss
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        #把正样本的分类损失置0
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)#将loss降序排列
        _, idx_rank = loss_idx.sort(1)#将loss的index值升序排列
        num_pos = pos.long().sum(1, keepdim=True)
        #num_neg<=self.negpos_ratio*num_pos
        num_neg = torch.clamp(self.negpos_ratio*num_pos + 200, max=pos.size(1)-1)#防止num_neg=0                                                    
        neg = idx_rank < num_neg.expand_as(idx_rank)#选出num_neg个负样本
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
```



先验框与ground truth的匹配

匹配原则:

1.找到与每个groundtruth 框 IOU最大的先验框,这样保证每个ground truth 都有先验框与之对应.

2.剩下的先验框若与某个ground truth的 IOU大于一个阈值也和该ground truth 匹配.如果说某个先验框和多个ground truth 框匹配,那么选IOU 最大那个.因为一个ground truth 可以对应多个先验框,但是一个先验框只能对应一个ground truth.

这样做的原因(个人理解):

1.如果只使用第一个原则,那么匹配到的正样本先验框的个数就等于ground truth 框的个数.正样本就太少了.

2.一个ground truth框 匹配一个先验框是不合理的.一个ground truth 框的周围会有很多的先验框,如果只使用一个先验框在训练的时候就比较难回归到这个框上.但是如果有多个先验框的话,网络的输出就容易匹配到其中一个.

3.一个priorbox  对应多个groudtruth.

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-174bec5acd695bacbdaa051b730f998a_720w.jpg)

```python

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index 即 IOU
    # shape:[num_groundtruth,num_prior]
    # 计算每一个priorbox和groundtruth的交并比
    #假如overlaps是这样的,有3个ground truth box,10个prior box.
    '''[[0.81, 0.63, 0.24, 0.10, 0.26, 0.26, 0.98, 0.18, 0.41, 0.44],
        [0.15, 0.80, 0.93, 0.85, 0.91, 0.13, 0.06, 0.69, 0.95, 0.19],
        [0.31, 0.84, 0.12, 0.86, 0.90, 0.14, 0.84, 0.80, 0.85, 0.45]] 
         '''
    overlaps =jaccard (
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    '''
    以ground truth box视角求最大值,即按行求最大值.
    best_prior_overlap: 
       [[0.98],
        [0.95],
        [0.90]]
    best_prior_idx: 
       [[6],
        [8],
        [4]]
    '''
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # ignore hard gt
    #排除与groundtruth<0.2的先验框
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    '''
    best_prior_idx_filter:
       [[6],
        [8],
        [4]]
    '''
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    #如果ground truth 为背景.
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    '''
    best_truth_overlap:
    [[0.81, 0.84, 0.93, 0.86, 0.91, 0.26, 0.98, 0.80, 0.95,0.45]]
    best_truth_idx: 
    [[0, 2, 1, 2, 1, 0, 0, 2, 1, 2]]
    '''
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    #先把使用第一原则匹配的先验框的IOU值置为一个较大的值,防止下面进行阈值过滤的时候给过滤掉,因为必须保证一#个groundtruth最少有一个先验框匹配.
    '''
    best_truth_overlap: 
    [0.81, 0.84, 0.93, 0.86, 2.00, 0.26, 2.00, 0.80, 2.00,0.45]
    '''
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    '''
    best_truth_idx: [0, 2, 1, 2, 2, 0, 0, 2, 1, 2]
    '''
    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j   #将第一原则匹配的框,合并到第二原则中.
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    '''
       [[0.89, 0.73, 0.71, 0.56],
        [0.60, 0.94, 0.87, 0.43],
        [0.66, 0.73, 0.85, 0.04],
        [0.60, 0.94, 0.87, 0.43],
        [0.60, 0.94, 0.87, 0.43],
        [0.89, 0.73, 0.71, 0.56],
        [0.89, 0.73, 0.71, 0.56],
        [0.60, 0.94, 0.87, 0.43],
        [0.66, 0.73, 0.85, 0.04],
        [0.60, 0.94, 0.87, 0.43]]
    '''
    conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm
```

softmax 损失函数

```python
 loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
 
 def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

```

softmax损失是交叉熵和softmax的结合

对于一个单样本的情况:

交叉熵:$L_{1}=-\sum_{j=1}^{C} y_{j} \log p_{j}$ 

对于一个二分类任务来说,C=2,设当前样本X1属于类A,则y=(1,0),设P=(0.9,0.1),则交叉熵为:

$L=-\left(1 * \log p_{1}+0 * \log p_{2}\right)=-\log p_{1}=-\log 0.9$

这里可以看到,实际上只有当前样本X1对应的类计算出来是有值的 (1*logp1),其他的项都是0,扩展到多分类也一样,

所以交叉熵公式可以简化为:

$L=-\log p_{i}$

对于一个批次的样本则为: $L=\frac{1}{n} \sum_{i=1}^{n}\left(-\log p_{i}\right)$ 

那么softmax loss 是什么呢?

$L=\frac{1}{n} \sum_{i=1}^{n}\left(-\log \frac{e^{p_{i}}}{\sum_{j=1}^{C} e^{p_{j}}}\right)$

很简单,就是用交叉熵包裹softmax激活函数.

展开一下:
$$
\begin{aligned}
\log \left(\frac{e^{x_{j}}}{\sum_{i=1}^{n} e^{x_{i}}}\right) &=\log \left(e^{x_{j}}\right)-\log \left(\sum_{i=1}^{n} e^{x_{i}}\right) \\
&=x_{j}-\log \left(\sum_{i=1}^{n} e^{x_{i}}\right)
\end{aligned}
$$
一个小trick:
$$
\begin{aligned}
\log \operatorname{Sum} \operatorname{Exp}\left(x_{1} \ldots x_{n}\right) &=\log \left(\sum_{i=1}^{n} e^{x_{i}}\right) \\
&=\log \left(\sum_{i=1}^{n} e^{x_{i}-c} e^{c}\right) \\
&=\log \left(e^{c} \sum_{i=1}^{n} e^{x_{i}-c}\right) \\
&=\log \left(\sum_{i=1}^{n} e^{x_{i}-c}\right)+\log \left(e^{c}\right) \\
&=\log \left(\sum_{i=1}^{n} e^{x_{i}-c}\right)+c
\end{aligned}
$$
这么做的目的是为了防止指数爆炸.

则 softmax loss 为

$-\log (\operatorname{Softmax}())=\log \left(\sum_{i=1}^{n} e^{x_{i}-c}\right)+c-x_{j}$ 

即代码:

```python
loss_c = log_sum_exp(batch_conf)- batch_conf.gather(1, conf_t.view(-1,1))
```

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-bbd4481c30a8d0d6a4e9dccff4c3e617_720w.jpg)



widerface.py

```python
class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations

        '''#给无标签数据打上全为0的标签
            annotations=np.append(annotations,np.zeros((1,15)),axis=0)
            target = np.array(annotations)
        '''
        
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target
```



### Anchor Settings



