import math
from builtins import range, print
import numpy as np
import cv2

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
        h, w = self.image.shape[0], self.image.shape[1]
        new_img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                new_img[i, j] = c * (math.log(1.0 + self.image[i, j]))
        new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
        new_img = np.uint8(new_img)
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
        h, w = self.image.shape[0], self.image.shape[1]
        new_img = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                new_img[i, j] = c * math.pow(self.image[i, j], gamma)
        cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.convertScaleAbs(new_img)
        return new_img

    def histogram_equalization(self, level=256):
        """
        直方图均衡化(HE)

        本函数在一个流程里用 for 循环完整展示 HE 的三步，不拆成别的函数调用：
        1) 遍历每个像素，统计各灰度级出现次数（直方图）；
        2) 从左到右累加直方图，得到累计分布 CDF，再换算成 0~(level-1) 的映射灰度；
        3) 再次遍历每个像素，用「原灰度 -> 新灰度」查表写出结果图。

        离散公式（与教材一致）：
        设总像素数 N = 行数×列数，灰度 r_k 的频数为 h(k)，则
            cdf(k) = h(0)+h(1)+...+h(k)
            新灰度 s_k = round((level-1) * cdf(k) / N)
        这里用整型累计，最后一步用 uint8 截断到合法范围。
        """
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
    def rgb_to_gray(self,save_root):
        gray_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_root, gray_img)
        self.image = gray_img

if __name__ == "__main__":
    # 将RGB转化为Gray
    img = cv2.imread("./image/Venti.png")  # 替换成你的图片路径
    grayscale_transformation = Grayscale_transformation(img)
    grayscale_transformation.rgb_to_gray("./image/Venti_gray.jpg")

    # 图像反转
    reversed_img = grayscale_transformation.reverse()
    cv2.imwrite('./image/Venti_reversed.jpg', reversed_img)

    # 对数变换
    # 把小数值拉开，把大数值压缩 → 暗部变亮，细节显现。
    log_img = grayscale_transformation.logTransform(60)
    cv2.imwrite('./image/Venti_log.jpg', log_img)

    # 伽马变换
    gamma_img = grayscale_transformation.gammaTransform(c = 60,gamma = 0.8)
    cv2.imwrite('./image/Venti_gamma.jpg', gamma_img)

    # 直方图转换
    histogram_equalization_img = grayscale_transformation.histogram_equalization()
    cv2.imwrite('./image/Venti_histogram_equalization.jpg', histogram_equalization_img)
