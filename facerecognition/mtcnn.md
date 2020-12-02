##                                                    mtcnn详解

### 主要工作

1. 采用三级级联卷积神经网络架构
2. 多任务学习:联合人脸检测框和人脸关键点进行学习
3. 提出一种在线难例挖掘策略  

### 实验结果

#### **人脸检测**

| ![image-20201110140821306](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201110140821306.png) | ![image-20201110140925629](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201110140925629.png) | ![image-20201110141111114](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201110141111114.png) |
| ------------------------------------------------------------ | :----------------------------------------------------------: | ------------------------------------------------------------ |
|                                                              |               **在widerface数据集上的pr曲线**                |                                                              |

#### **人脸对齐**

![image-20201111201401030](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201111201401030.png)

评价标准:the inter-ocular distance normalized error
$$
e_{i}=\frac{\left\|\mathrm{x}_{i}-\mathrm{x}_{i}^{*}\right\|_{2}}{d_{I O D}}
$$
选取n张人脸求分子为两点之间的距离,分母为两眼中心的间距,目的是对距离做一个归一化屏蔽脸部大小问题.

#### **算法效率**

在2.60GHz CPU上16fps, 在GPU (Nvidia Titan Black)上99fps.(matlab 代码).

### 整体流程

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-8e2d0268a6105c44af8512fbf24a3321_720w.jpg" alt="img" style="zoom: 80%;" />

1. **Image Pyramid:**制作图像金字塔（将尺寸从大到小的图像堆叠在一起类似金字塔形状，故名图像金字塔），对输入图像resize 到不同尺寸，输入网络。
2. **Stage 1 :** 将金字塔图像输入P-Net（Proposal Network)，获取含人脸的Proposal boundding boxes，并通过非极大值抑制（NMS）算法(后面对NMS算法的过程进行了补充）去除冗余框，这样便初步得到一些人脸检测候选框。
3. **Stage 2 :** 将P-Net输出得到的人脸图像输入R-Net（Refinement Network)，对人脸检测框坐标进行进一步的细化，通过NMS算法去除冗余框，此时的到的人脸检测框更加精准且冗余框更少。
4. **Stage 3 :** 将R-Net输出得到的人脸图像输入O-Net（Output Network)，一方面对人脸检测框坐标进行进一步的细化，另一方面输出人脸5个关键点（左眼、右眼、鼻子、左嘴角、右嘴角）坐标。

### 图像金字塔

- **图像金字塔的作用**

  因为待测试的图像中人脸的大小是不确定的，为了获取到包含合适人脸候选框，不同尺度的输入图像.是为了解决目标检测过程中目标尺度变化的问题。其缺点是图像的预处理过程比较耗时，特别是当输入图片较大时更耗时，从而影响推理速度。

  [解决目标多尺度问题的几种方法](https://zhuanlan.zhihu.com/p/92005927)

  1. 图像金字塔：图像处理效率太低
  2. 单个高层特征图：丧失了低层特征，不利于小目标的检测
  3. 多层特征：没有将高层语义特征和低层结构特征进行融合
  4. FPN：融合了高层语义特征和低层结构特征

- **图像金字塔的实现**

  1. 金字塔要建多少层，即一共要生成多少张图像

  2. 每张图像的尺寸如何确定
     在人脸检测时，通常要设置原图中要检测的最小人脸尺寸，原图中小于这个尺寸的人脸可以忽略，MTCNN代码中为minsize=20，MTCNN P-Net用于检测12×12大小的人脸。
     （1）人脸检测中的图像金字塔构建，涉及如下数据：
     输入图像尺寸，定义为（h, w） = （100, 120）
     最小人脸尺寸，定义为 min_face_size = 20
     最大人脸尺寸，如果不设置，为图像高宽中较短的那个，定义为max_face_size = 100
     网络能检测的人脸尺寸，定义为net_face_size = 12
     金字塔层间缩放比率，定义为factor = 0.709
     图像金字塔中
     最大缩放尺度max_scale = net_face_size / min_face_size = 12 / 20，
     最小缩放尺度min_scale = net_face_size / max_face_size = 12 / 100，
     中间的缩放尺度scale_n = max_scale * (factor ^ n) = ，
     对应的图像尺寸为(h_n, w_n) = (h * scale_n, w_n * scale_n)
     （2）举例解释说明：
     当原图（100, 120）中有一个最小人脸（20, 20）时，
     （20, 20）>（12, 12）图像缩小比（为缩放尺寸的倒数）最小20 / 12，原图由（100, 120）>（60, 72），图像（60, 72）为图像金字塔的最底层。根据金字塔层间缩放比率，每层图像的尺寸为：
     (h_n, w_n) = (h * scale_n, w_n * scale_n)
     =（100×12/20×0.709^ n，120×12/20×0.709^ n）
     同时需要保证min(h_n, w_n) >net_face_size

  3. 代码：

     ```python
         def processed_image(img, scale):
             height, width, channels = img.shape
             new_height = int(height * scale)  
             new_width = int(width * scale)  
             new_dim = (new_width, new_height)
             img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
             img_resized = (img_resized - 127.5) / 128   
             return img_resized
         
         def pyramid_image(img):
             net_size = 12       
             min_face_size = 20
             current_scale = float(net_size) / self.min_face_size  # find initial scale
             im_resized = processed_image(img, current_scale)  # the first layer of image pyramid
             current_height, current_width, _ = im_resized.shape
             while min(current_height, current_width) > net_size:
                 current_scale *= self.scale_factor
                 im_resized = processed_image(im, current_scale)
     ```

### 训练数据的生成

三个网络训练过程中的数据可分为4类样本数据，分别对应mtcnn的三种损失函数：

（1）face/non-face classification ：pos和neg样本数据

（2）bounding box regression：pos和part样本数据

（3 ) facial landmark localization：landmark样本数据

- **pos,net,part数据的生成**：

  其中，pos,net,part训练数据是根据gt的IOU大小决定的。

  - 对于pos样本，先根据gt，并产生随机偏移，iou和gt大于0.65被选为pos样本
  - 对于part样本，同样根据gt，并产生随机偏移，iou和gt在0.4～0.65被选为part样本
  - 对于neg样本，则是在图片随机crop， iou和gt小于0.3的被选为neg样本

  代码：

  ```python
  for annotation in annotations:
      annotation = annotation.strip().split(' ')
      #image path
      im_path = annotation[0]
      #print(im_path)
      #boxed change to float type
      bbox = list(map(float, annotation[1:]))
      #gt
      boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
      #load image
      img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
      idx += 1
      #if idx % 100 == 0:
          #print(idx, "images done")
  
      height, width, channel = img.shape
  
      neg_num = 0
      #1---->50
      # keep crop random parts, until have 50 negative examples
      # get 50 negative sample from every image
      while neg_num < 50:
          #neg_num's size [40,min(width, height) / 2],min_size:40
          # size is a random number between 12 and min(width,height)
          size = npr.randint(12, min(width, height) / 2)
          #top_left coordinate
          nx = npr.randint(0, width - size)
          ny = npr.randint(0, height - size)
          #random crop
          crop_box = np.array([nx, ny, nx + size, ny + size])
          #calculate iou
          Iou = IoU(crop_box, boxes)
  
          #crop a part from inital image
          cropped_im = img[ny : ny + size, nx : nx + size, :]
          #resize the cropped image to size 12*12
          resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
  
          if np.max(Iou) < 0.3:
              # Iou with all gts must below 0.3
              save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
              f2.write("../../DATA/12/negative/%s.jpg"%n_idx + ' 0\n')
              cv2.imwrite(save_file, resized_im)
              n_idx += 1
              neg_num += 1
              
      #for every bounding boxes
      for box in boxes:
          # box (x_left, y_top, x_right, y_bottom)
          x1, y1, x2, y2 = box
          #gt's width
          w = x2 - x1 + 1
          #gt's height
          h = y2 - y1 + 1
          # ignore small faces and those faces has left-top corner out of the image
          # in case the ground truth boxes of small faces are not accurate
          if max(w, h) < 20 or x1 < 0 or y1 < 0:
              continue      
          # crop another 5 images near the bounding box if IoU less than 0.5, save as negative samples
          for i in range(5):
              #size of the image to be cropped
              size = npr.randint(12, min(width, height) / 2)
              # delta_x and delta_y are offsets of (x1, y1)
              # max can make sure if the delta is a negative number , x1+delta_x >0
              # parameter high of randint make sure there will be intersection between bbox and cropped_box
              delta_x = npr.randint(max(-size, -x1), w)
              delta_y = npr.randint(max(-size, -y1), h)
              # max here not really necessary
              nx1 = int(max(0, x1 + delta_x))
              ny1 = int(max(0, y1 + delta_y))
              # if the right bottom point is out of image then skip
              if nx1 + size > width or ny1 + size > height:
                  continue
              crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
              Iou = IoU(crop_box, boxes)
      
              cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
              #rexize cropped image to be 12 * 12
              resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
      
              if np.max(Iou) < 0.3:
                  # Iou with all gts must below 0.3
                  save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                  f2.write("../../DATA/12/negative/%s.jpg" % n_idx + ' 0\n')
                  cv2.imwrite(save_file, resized_im)
                  n_idx += 1
  
          #产生positive和part
          #generate positive examples and part faces
          for i in range(20):
              # pos and part face size [minsize*0.8,maxsize*1.25]
              size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
  
              # delta here is the offset of box center
              if w<5:
                  print (w)
                  continue
              #print (box)
              delta_x = npr.randint(-w * 0.2, w * 0.2)
              delta_y = npr.randint(-h * 0.2, h * 0.2)
  
              #show this way: nx1 = max(x1+w/2-size/2+delta_x)
              # x1+ w/2 is the central point, then add offset , then deduct size/2
              # deduct size/2 to make sure that the right bottom corner will be out of
              nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
              #show this way: ny1 = max(y1+h/2-size/2+delta_y)
              ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
              nx2 = nx1 + size
              ny2 = ny1 + size
  
              if nx2 > width or ny2 > height:
                  continue 
              crop_box = np.array([nx1, ny1, nx2, ny2])
              #yu gt de offset
              offset_x1 = (x1 - nx1) / float(size)
              offset_y1 = (y1 - ny1) / float(size)
              offset_x2 = (x2 - nx2) / float(size)
              offset_y2 = (y2 - ny2) / float(size)
              #crop
              cropped_im = img[ny1 : ny2, nx1 : nx2, :]
              #resize
              resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
              box_ = box.reshape(1, -1)
              iou = IoU(crop_box, box_)
              if iou  >= 0.65:
                  save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                  f1.write("../../DATA/12/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                  cv2.imwrite(save_file, resized_im)
                  p_idx += 1
              elif iou >= 0.4:
                  save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                  f3.write("../../DATA/12/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                  cv2.imwrite(save_file, resized_im)
                  d_idx += 1
          box_idx += 1
          if idx % 100 == 0:
              print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
  ```

- **landmark数据的生成**

  原论文中使用的是CelebA数据集来生成的人脸关键点儿，tf版使用的是[这个数据集](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)，除了使用原始的标注信息，并对原始的groundtruth数据进行了随机的偏移，选取boundingbox IOU>0.65的数据用于训练，并做了mirror, roate, flip,anti-clockwise随机数据增强。

  代码：

  ```python
  def GenerateData(ftxt,data_path,net,argument=False):
      '''
      :param ftxt: name/path of the text file that contains image path,
                  bounding box, and landmarks
      :param output: path of the output dir
      :param net: one of the net in the cascaded networks
      :param argument: apply augmentation or not
      :return:  images and related landmarks
      '''
      if net == "PNet":
          size = 12
      elif net == "RNet":
          size = 24
      elif net == "ONet":
          size = 48
      else:
          print('Net type error')
          return
      image_id = 0
      #
      f = open(join(OUTPUT,"landmark_%s_aug.txt" %(size)),'w')
      #dstdir = "train_landmark_few"
      # get image path , bounding box, and landmarks from file 'ftxt'
      data = getDataFromTxt(ftxt,data_path=data_path)
      idx = 0
      #image_path bbox landmark(5*2)
      for (imgPath, bbox, landmarkGt) in data:
          #print imgPath
          F_imgs = []
          F_landmarks = []
          #print(imgPath)
          img = cv2.imread(imgPath)
  
          assert(img is not None)
          img_h,img_w,img_c = img.shape
          gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
          #get sub-image from bbox
          f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
          # resize the gt image to specified size
          f_face = cv2.resize(f_face,(size,size))
          #initialize the landmark
          landmark = np.zeros((5, 2))
  
          #normalize land mark by dividing the width and height of the ground truth bounding box
          # landmakrGt is a list of tuples
          for index, one in enumerate(landmarkGt):
              # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
              rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
              # put the normalized value into the new list landmark
              landmark[index] = rv
          
          F_imgs.append(f_face)
          F_landmarks.append(landmark.reshape(10))
          landmark = np.zeros((5, 2))        
          if argument:
              idx = idx + 1
              if idx % 100 == 0:
                  print(idx, "images done")
              x1, y1, x2, y2 = gt_box
              #gt's width
              gt_w = x2 - x1 + 1
              #gt's height
              gt_h = y2 - y1 + 1        
              if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                  continue
              #random shift
              for i in range(10):
                  bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                  delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                  delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                  nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x,0))
                  ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y,0))
  
                  nx2 = nx1 + bbox_size
                  ny2 = ny1 + bbox_size
                  if nx2 > img_w or ny2 > img_h:
                      continue
                  crop_box = np.array([nx1,ny1,nx2,ny2])
  
  
                  cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                  resized_im = cv2.resize(cropped_im, (size, size))
                  #cal iou
                  iou = IoU(crop_box, np.expand_dims(gt_box,0))
                  if iou > 0.65:
                      F_imgs.append(resized_im)
                      #normalize
                      for index, one in enumerate(landmarkGt):
                          rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                          landmark[index] = rv
                      F_landmarks.append(landmark.reshape(10))
                      landmark = np.zeros((5, 2))
                      landmark_ = F_landmarks[-1].reshape(-1,2)
                      bbox = BBox([nx1,ny1,nx2,ny2])                    
  
                      #mirror                    
                      if random.choice([0,1]) > 0:
                          face_flipped, landmark_flipped = flip(resized_im, landmark_)
                          face_flipped = cv2.resize(face_flipped, (size, size))
                          #c*h*w
                          F_imgs.append(face_flipped)
                          F_landmarks.append(landmark_flipped.reshape(10))
                      #rotate
                      if random.choice([0,1]) > 0:
                          face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                           bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                          #landmark_offset
                          landmark_rotated = bbox.projectLandmark(landmark_rotated)
                          face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                          F_imgs.append(face_rotated_by_alpha)
                          F_landmarks.append(landmark_rotated.reshape(10))
                  
                          #flip
                          face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                          face_flipped = cv2.resize(face_flipped, (size, size))
                          F_imgs.append(face_flipped)
                          F_landmarks.append(landmark_flipped.reshape(10))                
                      
                      #anti-clockwise rotation
                      if random.choice([0,1]) > 0: 
                          face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                           bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                          landmark_rotated = bbox.projectLandmark(landmark_rotated)
                          face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                          F_imgs.append(face_rotated_by_alpha)
                          F_landmarks.append(landmark_rotated.reshape(10))
                  
                          face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                          face_flipped = cv2.resize(face_flipped, (size, size))
                          F_imgs.append(face_flipped)
                          F_landmarks.append(landmark_flipped.reshape(10)) 
                      
              F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
              #print F_imgs.shape
              #print F_landmarks.shape
              for i in range(len(F_imgs)):
                  #if image_id % 100 == 0:
  
                      #print('image id : ', image_id)
  
                  if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                      continue
  
                  if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                      continue
  
                  cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)), F_imgs[i])
                  landmarks = map(str,list(F_landmarks[i]))
                  f.write(join(dstdir,"%d.jpg" %(image_id))+" -2 "+" ".join(landmarks)+"\n")
                  image_id = image_id + 1
      f.close()
      return F_imgs,F_landmark
  ```

  

### 三级级联网络架构

#### 第一级网络P-Net:

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201112180556206.png" alt="img" style="zoom: 50%;"/>

P-Net包含三个卷积层，每个卷积核的大小均为3x3，是一个全卷积网络,没有全连接层。

（1）作用：判断是否含人脸，并给出人脸框和关键点的位置，为O-Net提供人脸候选框。

（2）输入：训练输入尺寸大小为 12x12的三通道图像 

（3）输出：包含三部分：a.是否人脸的概率1x1x2向量；b.人脸检测框坐标（左上点和右下点）1x1x4向量；c.人脸关键点（5个关键点）坐标1x1x10向量。

#### 第二级网络R-Net:

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201112182909435.png" alt="image-20201112182909435" style="zoom:50%;" />

R-Net网络结构与P-Net的网络结构类似，也包含三个卷积层，前两个卷积核的大小均为3x3，第三个卷积核的大小为2x2，且其相比于P-Net 多了一个全连接层。

（1）作用：对P-Net 输出可能为人脸候选框图像进一步进行判定，同时细化人脸检测目标框精度。

（2）输入：训练输入尺寸大小为 24x24的三通道图像 

（3）输出：包含三部分：a.是否人脸的概率的1x1x2向量；b.人脸检测框坐标（左上点和右下点）1x1x4向量；c.人脸关键点坐标1x1x10向量。

#### 第三级网络O-Net:

<img src="https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20201112183317932.png" alt="image-20201112183317932" style="zoom:50%;" />

O-Net网络结构相比R-Net的网络结构，多了一个3x3卷积层，

（1）作用：对R-Net 输出可能为人脸的图像进一步进行判定，同时细化人脸检测目标框精度。

（2）输入：训练尺寸大小为 48x48的三通道图像 

（3）输出：包含三部分：a.是否人脸的概率的1x1x2向量；b.人脸检测框坐标（左上点和右下点）1x1x4向量；c.人脸关键点坐标1x1x10向量。

### 训练过程

- **损失函数**

由于在MTCNN 中，每个网络都有三个输出，因此会对应有三个损失函数：

1）对于输出a:是否人脸分类，由于是分类问题，其loss 函数使用常见的交叉熵损失：
$$
L_{i}^{\text {det}}=-\left(y_{i}^{\text {det}} \log \left(p_{i}\right)+\left(1-y_{i}^{\text {det}}\right)\left(1-\log \left(p_{i}\right)\right)\right)
$$
2）对于输出b:人脸检测框定位，由于是回归问题，所以使用L2 损失函数：
$$
L_{i}^{b o x}=\left\|\hat{y}_{i}^{b o x}-y_{i}^{b o x}\right\|_{2}^{2}
$$
3）对于输出c:人脸关键点定位，由于也是回归问题，所以也使用L2 损失函数：
$$
L_{i}^{\text {landmark}}=\left\|\hat{y}_{i}^{\text {landmark}}-y_{i}^{\text {landmark}}\right\|_{2}^{2}
$$
最终将三个损失函数进行加权累加，便得到总损失函数。由上述样本选择可知，当负样本和正样本训练时，由于仅用于分类，所以其仅有分类损失，而不存在人脸检测框和人脸关键点定位损失。即并不是所有的样本都同时存在上述三种损失。通过控制不同权重，使得三个网络的关注点不一样：
$$
\min \sum_{i=1}^{N} \sum_{j \in\{\text {det}, \text {box}, \text {landmark}\}} \alpha_{j} \beta_{i}^{j} L_{i}^{j}
$$
其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 表示权重值，对于P-Net 和R-Net 则更关注检测框定位的准确性，而较少关注关键点定位的准确性，所以关键点定位损失权重较小 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bdet%7D%3D1+%EF%BC%8C++%5Calpha_%7Bbox%7D%3D0.5+%EF%BC%8C%5Calpha_%7Blandmark%7D%3D0.5) ；而对于O-Net 则更关注关键点定位准确性，所以此时关键点定位损失权重较大 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bdet%7D%3D1+%EF%BC%8C++%5Calpha_%7Bbox%7D%3D0.5+%EF%BC%8C%5Calpha_%7Blandmark%7D%3D1) 。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 表示这样本对哪种loss有效，其取值范围为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta+%3D%5Cleft%28+0%2C1+%5Cright%29) ，比如当想使用负样本数据时， ![[公式]](https://www.zhihu.com/equation?tex=\beta^{det}%3D1+，\beta^{box}%3D0，\beta^{lamdmark}%3D0)

。这样就可以在训练过程中自由的选择用什么样的样本去训练什么样的loss,比如Negatives和Positives用于分类损失，Positives和Part用于回归人脸框，Landmark用于回归关键点儿。 

- **难样本挖掘**

不同于传统的难样本挖掘算法，论文中提出使用在线难样本挖掘。首先将前向传递得到的loss值降序排序，然后选择前70%大的样本作为难样本，并在反向传播的过程中只计算难样本的梯度，即将对于训练网络没有增强作用的简单样本忽略不计。

- **分阶段训练**
  1. 使用上面介绍训练数据的生成方法在widerface和celeba(或其他数据集)上分别生成positive,nagtive,part和landmark数据集。送入PNet进行训练。
  2. Rnet的训练pos，neg和part样本数据由先前训练好的Pnet产生，由Pnet对原始图片图片进行人脸检测，再根据预测的bbox与gt的iou值大小分别产生neg，pos和part数据，这是和Pnet训练数据产生差别之处。但它的landmark训练数据是从celeba单独产生的，只不过输入图片的大小是24*24.
  3. Onet训练数据和rnet类似，只是neg，part，pos的数据是经过前面训练的pnet，rnet产生，图片大小 为48*48



## 参考:

[论文链接:Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878.pdf) 

[mtcnn tensorflow版github](https://github.com/AITTSMD/MTCNN-Tensorflow)

[FPN以及其他几种解决多尺度问题的方法](https://zhuanlan.zhihu.com/p/92005927)

[mtcnn中构建图像金字塔](https://blog.csdn.net/krais_wk/article/details/101444330)

[人脸检测之mtcnn算法详解](https://zhuanlan.zhihu.com/p/63948672)

