# UD-Net
该项目首先将直肠癌的 CT 图像及其相应的 DCM 图像作为研究对象，并提出了将 U-NET 和 Densenet 结合的直肠肿瘤分割模型。该模型基于肿瘤特征信息，对直肠肿瘤区域做准确分割，实验精度达到了90％以上。此外，提出了基于3D卷积神经网络的直肠肿瘤分类模型，该模型以直肠肿瘤区域的淋巴结是否转移作为研究对象，构建肿瘤区域信息的三维张量，并获得直肠肿瘤的分类结果，分类精度达到65％。

## 数据预处理

本项目的实验数据由广州泰迪智能科技有限公司提供，是一个包含 107 个患者 DCM 格式的腹部横断位动脉期和门脉期两种增强 CT 影像的数据文件，大小为 2.88G。

本项目实验采用的实验环境：Google Colaboratory，Nvidia Tesla K80系列 GPU；操作系统为Windows 10，开发环境为 Python3.6。

本项目对原始数据集中的直肠肿瘤区域主要进行了图像纹理增强、数据增强等处理工作。CT 影像反映了人体器官和组织对 X 射线的吸收程度，但往往由于拍摄设备等因素，造成 CT 影像中存在大量“噪声”信息，为了加强图像的亮度对比和纹理细节，本项目对原始数据集进行直方图均衡和归一化处理：

-  直方图均衡化：通过对图像进行非线性拉伸，重新分配CT图像中的象元值，使一定灰度范围内象元值的数量大致相等，以均衡影像亮暗部对比。
-  归一化：保持原图结构不变，以减小 CT 图片由于光线不均对模型造成的影响。

以下是直方图均衡化和归一化的核心代码：

```python
#直方图均衡化：

clahe=cv2.createCLAHE(clipLimit=2.0,

tileGridSize=(4, 4))

img = clahe.apply(img)

#归一化：

MIN_BOUND = -1000.0

MAX_BOUND = 400.0

img = (img_mask[0] - MIN_BOUND) /

(MAX_BOUND - MIN_BOUND)

img[img_mask[0] > 1] = 1.

img[img_mask[0] < 0] = 0.



img=(img_mask[0]-

np.min(img_mask[0])) / (np.max(img_mask[0]) - np.min(img_mask[0]))

mask = img_mask[1] / 255.#
```

所使用的直方图均衡化采用局部均衡化，其中 clipLimit 代表对比度大小，tileGridSiz 代表处理的块大小。

​                                                          ![img](https://cdn.nlark.com/yuque/0/2019/png/323476/1556433311888-ddc95f6f-7617-4f9f-a83f-dad081feebdc.png)



医疗图像的特征远少于自然图像，本项目将直肠癌图像数目相对较少的种类，通过数据增强的方式，扩充数据集样本的数量。数据增强的核心代码如下所示。

```python
defunet_dataEnhance(self,img,mask,rotation_range=30,width_shift_range=0.05,height_shift_range=0.1, zoom_range=0.8, seed=5):
	img_list.append(self.shift(img, width_shift_range, randint=randint))
mask_list.append(self.shift(mask, width_shift_range, randint=randint))
new_img = np.zeros(img.shape)
for i in range(height):
   for j in range(width):
      if (j + w_shift) < 0 or (j + w_shift) >= width:
        continue
      new_img[i, j + w_shift] = img[i, j]
angle = random.randint(-rotation_range, rotation_range)
img_list.append(self.rotate_zoom(img, angle))
mask_list.append(self.rotate_zoom(mask, angle))
matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), angle, zoom_rate)
new_img = cv2.warpAffine(img, matRotate, (height, width))
```

其中，width_shift_range 和 height_shift_range 表示沿着水平或垂直的方向，将原始数据集中图像分别以宽和高的 0.05 和 0.1 倍进行平移；zoom_range=0.8 则表示将图片按照 0.8 的比例进行缩放；由于图片在进行旋转、平移操作时，图片信息会出现缺失，通过 warpAffine 函数默认的线性插值，使用用 0 将图片中缺失的信息进行填充。数据增强的结果如图所示。

![img](https://cdn.nlark.com/yuque/0/2019/png/323476/1556433674258-5fa3fac9-fa0e-4d99-8c5f-be4abe572624.png)
