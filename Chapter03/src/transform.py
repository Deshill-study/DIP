import math
from builtins import range, print
import numpy as np
import cv2

class Transform:
    def __init__(self,image):
        self.image = image
    
    def bitLayerTransform(image, layerNum)-> image:  
        if layerNum == 1:
            new_img = np.where((image >= 0) & (image < 2), 255, 0)
        elif layerNum == 2:
            new_img = np.where((image >= 2) & (image < 4), 255, 0)
        elif layerNum == 3:
            new_img = np.where((image >= 4) & (image < 8), 255, 0)
        elif layerNum == 4:
            new_img = np.where((image >= 8) & (image < 16), 255, 0)
        elif layerNum == 5:
            new_img = np.where((image >= 16) & (image < 32), 255, 0)
        elif layerNum == 6:
            new_img = np.where((image >= 32) & (image < 64), 255, 0)
        elif layerNum == 7:
            new_img = np.where((image >= 64) & (image < 128), 255, 0)
        elif layerNum == 8:
            new_img = np.where((image >= 128) & (image < 256), 255, 0)
        else:
            new_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            print('Please enter the number of bit layers from 1 to 8')

        return new_img.astype(np.uint8)

    def sliceTransform(image): # 灰度级分层
        # 二值映射
        """
        h, w = image.shape[0], image.shape[1]
        new_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if image[i, j] < 190 or image[i, j] > 230:
                    new_img[i, j] = 0
                else:
                    new_img[i, j] = 255
        """
        # 区域映射
        h, w = img.shape[0], img.shape[1]
        new_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if 230 >= img[i, j] >= 190:
                    new_img[i, j] = 255
                else:
                    new_img[i, j] = img[i, j]
        return new_img


    def logTransform(c, image): # 对数变换
        # 3通道RGB
        """
        h, w, d = image.shape[0], image.shape[1], image.shape[2]
        new_img = np.zeros((h, w, d))
        min = 255
        max = 0
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    new_img[i, j, k] = c * (math.log(1.0 + image[i, j, k]))

        # print(new_img)
        new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)/255.
        print(new_img.max(), new_img.min())
        print(new_img)

        return new_img
        """
        # 灰度图
        h, w = img.shape[0], img.shape[1]
        new_img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                new_img[i, j] = c * (math.log(1.0 + img[i, j]))
        new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)/255.
        return new_img
    def contrastStretchTransform(image): # 灰度拉伸变换
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
    def gammaTransform(c, gamma, image):
        # 彩色图像
        """
        h, w, d = image.shape[0], image.shape[1], image.shape[2]
        new_img = np.zeros((h, w, d), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    new_img[i, j, k] = c*math.pow(image[i, j, k], gamma)

        cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
        print(new_img)
        new_img = cv2.convertScaleAbs(new_img)
        print(new_img)
        """

        # 灰度图
        h, w = image.shape[0], image.shape[1]
        new_img = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                new_img[i, j] = c * math.pow(image[i, j], gamma)
        cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.convertScaleAbs(new_img)

        return new_img
    def reverse_img(self,image): #反转图像
        reverse_img = 255 - image
        return reverse_img
    

if __name__ == "__main__":
    transform = Transform()
