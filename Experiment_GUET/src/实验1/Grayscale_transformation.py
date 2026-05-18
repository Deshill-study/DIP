import math
from builtins import range, print
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Grayscale_transformation:
    def __init__(self,image):
        self.rgb_img = image
        self.image = None
    
    def reverse(self):
        new_image = 255 - self.image
        return new_image

    def bitLayerTransform(self, layerNum):  
        '''
        10000000(2^7)~11111111(2^8减1)
        '''
        if layerNum == 1:
            new_img = np.where((self.image >= 0) & (self.image < 2), 255, 0)
        elif layerNum == 2:
            new_img = np.where((self.image >= 2) & (self.image < 4), 255, 0)
        elif layerNum == 3:
            new_img = np.where((self.image >= 4) & (self.image < 8), 255, 0)
        elif layerNum == 4:
            new_img = np.where((self.image >= 8) & (self.image < 16), 255, 0)
        elif layerNum == 5:
            new_img = np.where((self.image >= 16) & (self.image < 32), 255, 0)
        elif layerNum == 6:
            new_img = np.where((self.image >= 32) & (self.image < 64), 255, 0)
        elif layerNum == 7:
            new_img = np.where((self.image >= 64) & (self.image < 128), 255, 0)
        elif layerNum == 8:
            new_img = np.where((self.image >= 128) & (self.image < 256), 255, 0)
        else:
            new_img = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            print('Please enter the number of bit layers from 1 to 8')

        return new_img.astype(np.uint8)

    def sliceTransform(self,s,e): # 灰度级分层
        # 二值映射
        """
        h, w = image.shape[0], image.shape[1]
        new_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if image[i, j] < s or image[i, j] > e:
                    new_img[i, j] = 0
                else:
                    new_img[i, j] = 255
        """
        # 区域映射
        h, w = self.image.shape[0], self.image.shape[1]
        new_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if e >= self.image[i, j] >= s:
                    new_img[i, j] = 255
                else:
                    new_img[i, j] = self.image[i, j]
        return new_img


    def logTransform(self,c): # 对数变换
        # 灰度图
        h, w = self.image.shape[0], self.image.shape[1]
        new_img = np.zeros((h, w),dtype=np.float32)
        for i in range(h):
            for j in range(w):
                new_img[i, j] = (math.log(1.0 + self.image[i, j]))

        new_img_norm = cv2.normalize(new_img, None, 0, 1, cv2.NORM_MINMAX)
        new_img = c*new_img_norm*255
        new_img = np.uint8(np.clip(new_img, 0, 255))
        return new_img
    def contrastStretchTransform(self,image): # 灰度拉伸变换
        """
        灰度拉伸
        定义: 灰度拉伸, 也称对比度拉伸, 是一种简单的线性点运算.
        作用: 扩展图像的直方图, 使其充满整个灰度等级范围内.
        公式: A = min[f(x, y)], 最小灰度级;
            B = max[f(x, y)], 最大灰度级;
            f(x, y)为输入图像, g(x, y)为输出图像.
        缺点: 如果灰度图像中最小值A=0, 最大值B=255, 则图像没有什么改变.
        """
        # 彩色图像
        h, w, d = image.shape[0], image.shape[1], image.shape[2]
        new_img = np.zeros((h, w, d), dtype=np.float32)
        A = image.min()
        B = image.max()
        print(A, B)
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    new_img[i, j, k] = 255.0 / (B - A) * (image[i, j, k] - A) + 0.5
        new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.convertScaleAbs(new_img)

        return new_img
    def gammaTransform(self,c, gamma):# 伽马变换
        # 灰度图
        h, w = self.image.shape[0], self.image.shape[1]
        new_img = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                new_img[i, j] = c * math.pow(self.image[i, j], gamma)
        cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.convertScaleAbs(new_img)
        return new_img

    def histogram_equalization(self, level=256):
        m, n = self.image.shape
        total_pixels = m * n

        # 1. 统计直方图
        hist = [0] * level
        for i in range(m):
            for j in range(n):
                g = int(self.image[i, j])
                if 0 <= g < level:
                    hist[g] += 1

        # 2. 计算前缀和
        cum = [0] * level
        running = 0
        for k in range(level):
            running += hist[k]
            cum[k] = running

        # 建立映射
        map_gray = [0] * level
        if total_pixels > 0:
            scale = float(level - 1) / float(total_pixels)
            for k in range(level):
                map_gray[k] = int(round(scale * cum[k]))
                if map_gray[k] > level - 1:
                    map_gray[k] = level - 1
                if map_gray[k] < 0:
                    map_gray[k] = 0

        # 映射到新的图像
        out = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                g = int(self.image[i, j])
                if 0 <= g < level:
                    out[i, j] = map_gray[g]
                else:
                    out[i, j] = 0
        return out
    def rgb_to_gray(self):
        gray_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY)
        self.image = gray_img
        return gray_img
    def BWBand(self, w, h):
        img = np.zeros([321, 201], dtype=np.uint8)
        # 计算中心坐标
        center_row = 321 // 2 
        center_col = 201 // 2 
        start_y = center_row - h // 2  # 上
        end_y = center_row + h // 2    # 下
        start_x = center_col - w // 2  # 左
        end_x = center_col + w // 2    # 右
        img[start_y:end_y, start_x:end_x] = 255
        return img
    def gray_layer_Transform(self,layers): # 灰度级转换
        imgs = []
        for layer in layers:
            step = 256//layer
            h, w = self.image.shape[0], self.image.shape[1]
            new_img = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    new_img[i, j] =(self.image[i,j]//step)*step
            imgs.append(new_img)
        return imgs
    def spatial_transform(self,scales):
        # 首先缩小的话像素点肯定变小，那么要用少的像素点去对应原图，那肯定每个像素点的位置要乘scale
        '''
        空间分辨率缩小 = 隔行隔列像素采样
        缩小 1/2：每 2 行 2 列 取 1 个像素
        缩小 1/4：每 4 行 4 列 取 1 个像素
        缩小 1/32：每 32 行 32 列 取 1 个像素
        '''
        new_imgs = []
        h,w = self.image.shape
        for scale in scales:
            new_h = h//scale
            new_w = w//scale
            new_img = np.zeros((new_h,new_w),dtype=np.uint8)
            for i in range(new_h):
                for j in range(new_w):
                    new_img[i,j] = self.image[i*scale,j*scale]
            new_imgs.append(new_img)
        return new_imgs
    def PDF(self,image,level=256):
        times = [0]*level
        h,w = image.shape
        for i in range(h):
            for j in range(w):
                times[image[i,j]]+=1
        return times
if __name__ == "__main__":
    # 将RGB转化为Gray
    img = cv2.imread("./image/Venti.png")  # 替换成你的图片路径
    grayscale_transformation = Grayscale_transformation(img)
    gray_img = grayscale_transformation.rgb_to_gray()
    plt.subplot(2,3,1)
    plt.imshow(gray_img,cmap='gray')
    plt.axis('off')
    plt.title('gray_img')

    # 图像反转
    reversed_img = grayscale_transformation.reverse()
    cv2.imwrite('./image/Venti_reversed.jpg', reversed_img)
    plt.subplot(2,3,2)
    plt.imshow(reversed_img,cmap='gray')
    plt.axis('off')
    plt.title('gray_reversed_img')
    
    # 对数变换
    # 把小数值拉开，把大数值压缩 → 暗部变亮，细节显现。
    log_img = grayscale_transformation.logTransform(40)
    cv2.imwrite('./image/Venti_log.jpg', log_img)
    plt.subplot(2,3,3)
    plt.imshow(log_img,cmap='gray')
    plt.axis('off')
    plt.title('log_transform_img')

    # 伽马变换
    gamma_img = grayscale_transformation.gammaTransform(c = 60,gamma = 0.8)
    cv2.imwrite('./image/Venti_gamma.jpg', gamma_img)
    plt.subplot(2,3,4)
    plt.imshow(gamma_img,cmap='gray')
    plt.axis('off')
    plt.title('gamma_transform_img')

    # 直方图转换
    histogram_equalization_img = grayscale_transformation.histogram_equalization()
    cv2.imwrite('./image/Venti_histogram_equalization.jpg', histogram_equalization_img)
    plt.subplot(2,3,5)
    plt.imshow(histogram_equalization_img,cmap='gray')
    plt.axis('off')
    plt.title('HE_transform_img')

    r,g,b = img[:,:,2],img[:,:,1],img[:,:,0]
    origin_img = np.dstack([r,g,b])
    # 原来的图
    plt.subplot(2,3,6)
    plt.imshow(origin_img)
    plt.axis('off')
    plt.title('origin_img')
    plt.show()

BWBand_img = grayscale_transformation.BWBand(7,11)
plt.imshow(BWBand_img,cmap='gray')
plt.axis('off')
plt.title('WBand_img')
plt.show()


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

gray_img = read_gray('/home/ubuntu/codebase/yexijia/github_project/DIP/Experiment_GUET/src/class_img/dark.jpg')

grayscale_transformation.image = gray_img

reversed_img = grayscale_transformation.reverse()
gray_layer_imgs = grayscale_transformation.gray_layer_Transform([256,128,64,32,16,8,4,2])
nums = [256,128,64,32,16,8,4,2]
plt.figure(figsize=(15,8))
for index,gray_layer_img in enumerate(gray_layer_imgs):
    plt.subplot(2,4,index+1)
    plt.imshow(gray_layer_img,cmap='gray')
    plt.axis('off')
    plt.title(f'2^{8-index} colors')
plt.show()


scales = [1,2,4,8,16,32]
gray_imgs = grayscale_transformation.spatial_transform(scales)

plt.figure(figsize=(15,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(gray_imgs[i], cmap='gray')
    plt.title(f'1/{scales[i]} size')
    plt.axis('off')
plt.show()

# 原图
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')
pdf = grayscale_transformation.PDF(grayscale_transformation.image)

# 原图直方图

plt.subplot(2,2,2)
plt.bar(range(256),pdf,width=1)
plt.title('origin_img_pdf_hist')

# 反转图

plt.subplot(2,2,3)
plt.imshow(reversed_img,cmap='gray')
plt.title('reversed_img')
plt.axis('off')
pdf = grayscale_transformation.PDF(reversed_img)

# 反转直方图
plt.subplot(2,2,4)
plt.bar(range(256),pdf,width=1)
plt.title('reversed_img_pdf_hist')
plt.show()

# 原图
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')
pdf = grayscale_transformation.PDF(grayscale_transformation.image)

# 原图直方图

plt.subplot(2,2,2)
plt.bar(range(256),pdf,width=1)
plt.title('origin_img_pdf_hist')


# 均衡化后的图
histogram_equalization_img = grayscale_transformation.histogram_equalization()
plt.subplot(2,2,3)
plt.imshow(histogram_equalization_img,cmap='gray')
plt.axis('off')
plt.title('HE_transform_img')
pdf = grayscale_transformation.PDF(histogram_equalization_img)


# 均衡化后的直方图
plt.subplot(2,2,4)
plt.bar(range(256),pdf,width=1)
plt.title('HE_transform_img_pdf_hist')
plt.show()

# 原图
plt.figure(figsize=(15,8))
plt.subplot(2,4,1)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')


# 伽马变换
for i,gamma in enumerate([0.1,0.3,0.4,0.5,3.0,4.0,5.0]):
    gamma_img = grayscale_transformation.gammaTransform(c = 1,gamma = gamma)
    plt.subplot(2,4,i+2)  
    plt.imshow(gamma_img,cmap='gray')
    plt.axis('off')
    plt.title(f's = t^{gamma}')
plt.show()

# 原图
plt.figure(figsize=(15,8))
plt.subplot(2,4,1)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')


# 对数变换
for i,times in enumerate([0.05,0.07,0.08,0.1,0.12,0.13,0.14]):
    log_img = grayscale_transformation.logTransform(times)
    plt.subplot(2,4,i+2)
    plt.imshow(log_img,cmap='gray')
    plt.axis('off')
    plt.title(f's = {times} * log10(1.0 + t)')
plt.show()

# 原图
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')

# 伽马变换
gamma_img = grayscale_transformation.gammaTransform(c = 1,gamma = 0.5)
plt.subplot(3,3,2)  
plt.imshow(gamma_img,cmap='gray')
plt.axis('off')
plt.title(f's = t^0.5')

#直方图
pdf = grayscale_transformation.PDF(gamma_img)
plt.subplot(3,3,3)
plt.bar(range(256),pdf,width=1)
plt.title('s = t^0.5_pdf_hist')


# 原图
plt.subplot(3,3,4)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')

# 对数变换
log_img = grayscale_transformation.logTransform(0.14)
plt.subplot(3,3,5)
plt.imshow(log_img,cmap='gray')
plt.axis('off')
plt.title(f's = {0.14} * log10(1.0 + t)')

# 直方图
pdf = grayscale_transformation.PDF(log_img)
plt.subplot(3,3,6)
plt.bar(range(256),pdf,width=1)
plt.title(f's = {0.14} * log10(1.0 + t)_pdf_hist')



# 原图
plt.subplot(3,3,7)
plt.imshow(grayscale_transformation.image,cmap='gray')
plt.title('origin_img')
plt.axis('off')


# 均衡化后的图
histogram_equalization_img = grayscale_transformation.histogram_equalization()
plt.subplot(3,3,8)
plt.imshow(histogram_equalization_img,cmap='gray')
plt.axis('off')
plt.title('HE_transform_img')
pdf = grayscale_transformation.PDF(histogram_equalization_img)


# 均衡化后的直方图
plt.subplot(3,3,9)
plt.bar(range(256),pdf,width=1)
plt.title('HE_transform_img_pdf_hist')
plt.show()