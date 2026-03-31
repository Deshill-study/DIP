"""
空间域滤波封装（与 Chapter03/3.4 空间滤波器 内容对应）。

依赖 OpenCV：彩色图按 BGR、uint8 约定处理，与教材配套示例一致。
"""
from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np

ImageArray = np.ndarray


class SpatialFilterProcessor:
    """
    空间滤波处理类

    将平滑（线性/统计排序）、锐化（拉普拉斯、梯度）及边缘检测（Canny）
    等常见空域操作集中到一个类中，便于实验与对比。

    说明:
        - 多数 OpenCV 滤波函数可直接处理多通道图像；统计排序类中的
          最大值/最小值滤波为教学用双重循环实现，彩色输入会先转为灰度，
          与仓库内 ``maxminblur.py`` 行为一致。
        - 涉及一阶导数（Sobel、Prewitt、Roberts）时，中间用 ``CV_16S``
          避免 uint8 截断，再 ``convertScaleAbs`` 转回可显示范围。
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 平滑 — 线性
    # ------------------------------------------------------------------

    def mean_blur(self, image: ImageArray, ksize: Tuple[int, int] = (3, 3)) -> ImageArray:
        """
        均值滤波（平滑线性滤波器）

        用邻域内像素灰度（或各通道）的平均值代替中心像素，起到模糊与抑噪作用；
        模板越大图像越平滑，边缘与细节越容易被抹平。
        """
        return cv2.blur(image, ksize)

    def box_blur(
        self,
        image: ImageArray,
        ksize: Tuple[int, int] = (3, 3),
        normalize: bool = True,
    ) -> ImageArray:
        """
        盒式滤波，与 ``mean_blur`` 在 normalize=True 时类似；
        ``normalize=False`` 时相当于邻域求和，可用于特定卷积实验。
        """
        return cv2.boxFilter(image, -1, ksize, normalize=1 if normalize else 0)

    def gaussian_blur(
        self,
        image: ImageArray,
        ksize: Tuple[int, int] = (3, 3),
        sigma_x: float = 0,
        sigma_y: float = 0,
    ) -> ImageArray:
        """
        高斯加权平滑。相比均匀均值，权值随距中心距离衰减，模糊更“自然”，
        常用于降噪预处理（例如 Canny 前先高斯平滑）。
        """
        return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x, sigmaY=sigma_y)

    # ------------------------------------------------------------------
    # 平滑 — 统计排序（非线性）
    # ------------------------------------------------------------------

    def median_blur(self, image: ImageArray, ksize: int = 3) -> ImageArray:
        """
        中值滤波：将邻域像素排序后取中位数赋给中心像素。

        对椒盐噪声特别有效，且比同样尺寸的均值滤波更能保留边缘的阶跃结构。
        ``ksize`` 必须为正奇数。
        """
        if ksize % 2 == 0 or ksize < 1:
            raise ValueError("median_blur 的 ksize 须为正奇数")
        return cv2.medianBlur(image, ksize)

    def max_filter(self, image: ImageArray, ksize: int = 3) -> ImageArray:
        """邻域内取最大值，有膨胀亮细节的效果，可抑制暗点噪声。"""
        return self._max_min_filter(image, ksize, mode="max")

    def min_filter(self, image: ImageArray, ksize: int = 3) -> ImageArray:
        """邻域内取最小值，有腐蚀暗区域的效果，可抑制亮点噪声。"""
        return self._max_min_filter(image, ksize, mode="min")

    def _max_min_filter(
        self, image: ImageArray, ksize: int, mode: Literal["max", "min"]
    ) -> ImageArray:
        """双重循环实现最大/最小值滤波；彩色图先转灰度。"""
        img = image.copy()
        if img.ndim == 3 and img.shape[2] == 3:
            work = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            work = img.copy()
        else:
            raise ValueError("仅支持单通道灰度或 BGR 三通道图像")

        rows, cols = work.shape
        padding = (ksize - 1) // 2
        padded = cv2.copyMakeBorder(
            work, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
        )
        out = work.copy()
        for i in range(rows):
            for j in range(cols):
                roi = padded[i : i + ksize, j : j + ksize]
                min_val, max_val, _, _ = cv2.minMaxLoc(roi)
                if mode == "max":
                    out[i, j] = max_val
                else:
                    out[i, j] = min_val
        return out

    # ------------------------------------------------------------------
    # 锐化 — 拉普拉斯（二阶微分）
    # ------------------------------------------------------------------

    def laplacian(self, image: ImageArray, ksize: int = 3) -> ImageArray:
        """
        拉普拉斯算子：对图像做二阶微分近似，响应与灰度突变程度相关，
        突出边缘与细节，同时会放大噪声。输出为取绝对值后的 uint8，便于显示。
        """
        lap = cv2.Laplacian(image, cv2.CV_16S, ksize=ksize)
        return cv2.convertScaleAbs(lap)

    def laplacian_sharpen(
        self,
        image: ImageArray,
        ksize: int = 3,
        weight: float = 1.0,
    ) -> ImageArray:
        """
        拉普拉斯锐化叠加：g = f + c * Laplacian(f) 的离散近似。

        ``weight`` 为叠加强度 c；先将原图与拉普拉斯结果转为 float 相加，
        再裁剪到 [0,255] 并转 uint8，避免溢出。
        """
        lap = cv2.Laplacian(image, cv2.CV_16S, ksize=ksize)
        lap_u8 = cv2.convertScaleAbs(lap)
        base = image.astype(np.float32)
        detail = lap_u8.astype(np.float32)
        sharp = np.clip(base + weight * detail, 0, 255).astype(np.uint8)
        return sharp

    # ------------------------------------------------------------------
    # 锐化 / 边缘 — 一阶微分（梯度）
    # ------------------------------------------------------------------

    def sobel_gradient(
        self,
        image: ImageArray,
        ksize: int = 3,
        combine: Literal["sum", "max"] = "sum",
    ) -> ImageArray:
        """
        Sobel 梯度：分别求 x、y 方向导数，再合成幅值。

        中间使用 ``CV_16S`` 防止求导后越界被截断；``combine='sum'`` 时用
        加权平均（0.5, 0.5），与仓库内 ``Sobel.py`` 一致；``'max'`` 为逐点取较大绝对响应。
        """
        gx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=ksize)
        gy = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=ksize)
        abs_x = cv2.convertScaleAbs(gx)
        abs_y = cv2.convertScaleAbs(gy)
        if combine == "sum":
            return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        return cv2.max(abs_x, abs_y)

    def prewitt_gradient(self, image: ImageArray) -> ImageArray:
        """
        Prewitt 算子：3×3 一阶差分近似梯度，对噪声略平滑于 Roberts。
        """
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        return self._gradient_from_kernels(image, kx, ky)

    def roberts_gradient(self, image: ImageArray) -> ImageArray:
        """
        Roberts 交叉梯度：2×2 模板，对 45°/135° 方向边缘较敏感，实现简单。
        """
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        return self._gradient_from_kernels(image, kx, ky)

    def _gradient_from_kernels(
        self, image: ImageArray, kx: np.ndarray, ky: np.ndarray
    ) -> ImageArray:
        """对多通道图逐通道 filter2D，再合成梯度幅值（与 Sobel 合成方式一致）。"""
        if image.ndim == 2:
            gx = cv2.filter2D(image, cv2.CV_16S, kx)
            gy = cv2.filter2D(image, cv2.CV_16S, ky)
            return cv2.addWeighted(
                cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0
            )
        b, g, r = cv2.split(image)
        out_b = self._gradient_from_kernels(b, kx, ky)
        out_g = self._gradient_from_kernels(g, kx, ky)
        out_r = self._gradient_from_kernels(r, kx, ky)
        return cv2.merge([out_b, out_g, out_r])

    # ------------------------------------------------------------------
    # 边缘检测 — Canny
    # ------------------------------------------------------------------

    def canny(
        self,
        image: ImageArray,
        threshold1: float = 50,
        threshold2: float = 150,
        aperture_size: int = 3,
        blur_ksize: Tuple[int, int] = (3, 3),
        blur_sigma: float = 0,
    ) -> ImageArray:
        """
        Canny 边缘检测：内部先转灰度，可选高斯平滑，再双阈值滞后阈值化。

        仅输出单通道边缘图（uint8，0 或 255）。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if blur_ksize[0] > 0 and blur_ksize[1] > 0:
            gray = cv2.GaussianBlur(gray, blur_ksize, blur_sigma)
        return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)

    # ------------------------------------------------------------------
    # 统一入口（可选）
    # ------------------------------------------------------------------

    def apply(
        self,
        image: ImageArray,
        method: str,
        **kwargs,
    ) -> ImageArray:
        """
        按名称调用上述方法，便于脚本里字符串配置。

        method 取值示例:
            mean, box, gaussian, median, max, min,
            laplacian, laplacian_sharpen, sobel, prewitt, roberts, canny
        """
        name = method.lower().strip()
        dispatch = {
            "mean": lambda: self.mean_blur(image, kwargs.get("ksize", (3, 3))),
            "box": lambda: self.box_blur(
                image,
                kwargs.get("ksize", (3, 3)),
                kwargs.get("normalize", True),
            ),
            "gaussian": lambda: self.gaussian_blur(
                image,
                kwargs.get("ksize", (3, 3)),
                kwargs.get("sigma_x", 0),
                kwargs.get("sigma_y", 0),
            ),
            "median": lambda: self.median_blur(image, kwargs.get("ksize", 3)),
            "max": lambda: self.max_filter(image, kwargs.get("ksize", 3)),
            "min": lambda: self.min_filter(image, kwargs.get("ksize", 3)),
            "laplacian": lambda: self.laplacian(image, kwargs.get("ksize", 3)),
            "laplacian_sharpen": lambda: self.laplacian_sharpen(
                image, kwargs.get("ksize", 3), kwargs.get("weight", 1.0)
            ),
            "sobel": lambda: self.sobel_gradient(
                image, kwargs.get("ksize", 3), kwargs.get("combine", "sum")
            ),
            "prewitt": lambda: self.prewitt_gradient(image),
            "roberts": lambda: self.roberts_gradient(image),
            "canny": lambda: self.canny(
                image,
                kwargs.get("threshold1", 50),
                kwargs.get("threshold2", 150),
                kwargs.get("aperture_size", 3),
                kwargs.get("blur_ksize", (3, 3)),
                kwargs.get("blur_sigma", 0),
            ),
        }
        if name not in dispatch:
            raise ValueError(f"未知 method: {method}")
        return dispatch[name]()
