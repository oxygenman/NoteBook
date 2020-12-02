# Retinaface详解

## 主要工作

## 实验结果

## 实现细节

#### Multibox loss的实现

$$
\begin{aligned}
L &=L_{c l s}\left(p_{i}, p_{i}^{*}\right)+\lambda_{1} p_{i}^{*} L_{b o x}\left(t_{i}, t_{i}^{*}\right)+\lambda_{2} p_{i}^{*} L_{p t s}\left(l_{i}, l_{i}^{*}\right)+\lambda_{3} p_{i}^{*} L_{p i x e l}
\end{aligned}
$$

Retinaface中Multi-task Loss 分为四个部分:

(1)  $L_{c l s}\left(p_{i}, p_{i}^{*}\right):$  为softmax损失函数,即先算softmax,再算交叉熵.其中$p_i$为第i个priorbox是人脸的概率,$p_{i}^*$为groundtruth类别标签,当为人脸时为1,不是人脸时是0.

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
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
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





### Anchor Settings



