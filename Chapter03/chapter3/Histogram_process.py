import numpy as np
from PIL import Image, ImageOps


class HistogramProcessor:
    """
    直方图处理类

    该类封装了常见的图像对比度增强方法：
    1. HE    : 全局直方图均衡化
    2. AHE   : 自适应直方图均衡化
    3. CLAHE : 对比度受限的自适应直方图均衡化
    4. BRIGHT: 按亮度分段后分别均衡
    """

    def enhance_contrast(
        self,
        img,
        method="HE",
        level=256,
        window_size=32,
        affect_size=16,
        blocks=8,
        threshold=10.0,
    ):
        """
        统一增强入口，根据 method 选择算法。

        参数:
            img: PIL.Image 或 numpy.ndarray
            method: 算法名称，支持 HE/AHE/CLAHE/STANDARD/BRIGHT
            level: 灰度级数量，默认 256
            window_size: AHE 中统计窗口大小
            affect_size: AHE 中每次映射实际影响的区域大小
            blocks: CLAHE 中分块数量(行和列各分 blocks 块)
            threshold: CLAHE 的裁剪阈值倍数
        返回:
            PIL.Image，增强后的图像
        """
        # 统一转换为 numpy 数组，便于后续处理
        img_arr = np.array(img)

        # 根据 method 选择函数
        method_u = str(method).upper()
        if method_u in ("HE", "FHE"):
            func = self.histogram_equalization
        elif method_u == "AHE":
            func = self.adaptive_histogram_equalization
        elif method_u in ("CLAHE", "CLANE"):
            func = self.contrast_limited_adaptive_histogram_equalization
        elif method_u == "STANDARD":
            func = self.standard_histogram_equalization
        elif method_u in ("BRIGHT", "BRIGHT_LEVEL"):
            func = self.bright_wise_histogram_equalization
        else:
            raise ValueError(f"未知 method: {method}")

        # 灰度图: 直接处理
        if img_arr.ndim == 2:
            out = func(
                img_arr,
                level=level,
                window_size=window_size,
                affect_size=affect_size,
                blocks=blocks,
                threshold=threshold,
            )
            return Image.fromarray(out)

        # 彩色图: 分通道处理后再合并，仅处理 RGB 三个通道
        if img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
            channels = []
            for c in range(3):
                ch = func(
                    img_arr[:, :, c],
                    level=level,
                    window_size=window_size,
                    affect_size=affect_size,
                    blocks=blocks,
                    threshold=threshold,
                )
                channels.append(Image.fromarray(ch))
            return Image.merge("RGB", tuple(channels))

        raise ValueError("输入图像维度不支持，必须是灰度图或 RGB/RGBA 图像")

    def histogram_equalization(self, img_arr, level=256, **kwargs):
        """
        全局直方图均衡化(HE)

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
        m, n = img_arr.shape
        total_pixels = m * n

        # 1. 统计直方图
        hist = [0] * level
        for i in range(m):
            for j in range(n):
                g = int(img_arr[i, j])
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
                g = int(img_arr[i, j])
                if 0 <= g < level:
                    out[i, j] = map_gray[g]
                else:
                    out[i, j] = 0

        return out

    def adaptive_histogram_equalization(
        self, img_arr, level=256, window_size=32, affect_size=16, **kwargs
    ):
        """
        自适应直方图均衡化(AHE)

        与全局 HE 的区别:
        - HE 使用整幅图一个映射函数
        - AHE 在局部窗口内计算映射函数，从而更好增强局部细节

        参数含义:
        - window_size: 计算局部映射的窗口大小
        - affect_size: 每个窗口映射后真正覆盖回输出图的中心区域
        """
        out = img_arr.copy()
        m, n = img_arr.shape

        # 计算行列方向需要遍历多少个窗口
        rows = int((m - window_size) / affect_size + 1)
        cols = int((n - window_size) / affect_size + 1)
        if (m - window_size) % affect_size != 0:
            rows += 1
        if (n - window_size) % affect_size != 0:
            cols += 1

        # 中心影响区域相对窗口边界的偏移
        off = int((window_size - affect_size) / 2)

        for i in range(rows):
            for j in range(cols):
                # 影响区域边界
                asi, aei = i * affect_size + off, (i + 1) * affect_size + off
                asj, aej = j * affect_size + off, (j + 1) * affect_size + off

                # 窗口边界
                wsi, wei = i * affect_size, i * affect_size + window_size
                wsj, wej = j * affect_size, j * affect_size + window_size

                # 边界裁剪，防止超出数组范围
                wsi, wei = max(0, wsi), min(m, wei)
                wsj, wej = max(0, wsj), min(n, wej)
                asi, aei = max(0, asi), min(m, aei)
                asj, aej = max(0, asj), min(n, aej)

                window = img_arr[wsi:wei, wsj:wej]
                block = self.histogram_equalization(window, level=level)

                # 将均衡后的中心区域写回输出图像
                out[
                    asi:aei,
                    asj:aej,
                ] = block[
                    (asi - wsi):(aei - wsi),
                    (asj - wsj):(aej - wsj),
                ]
        return out.astype(np.uint8)

    def contrast_limited_adaptive_histogram_equalization(
        self, img_arr, level=256, blocks=8, threshold=10.0, **kwargs
    ):
        """
        对比度受限的自适应直方图均衡化(CLAHE)

        核心思想:
        1) 把图像分块，每块计算自己的直方图映射
        2) 对每块直方图做 clip，避免噪声被过分放大
        3) 对像素使用周围映射做插值，减轻块效应
        """
        m, n = img_arr.shape
        block_m = max(1, int(m / blocks))
        block_n = max(1, int(n / blocks))

        # 预计算每个块的 CDF 映射表
        maps = []
        for i in range(blocks):
            row_maps = []
            for j in range(blocks):
                si, ei = i * block_m, min(m, (i + 1) * block_m)
                sj, ej = j * block_n, min(n, (j + 1) * block_n)

                block_img = img_arr[si:ei, sj:ej]
                if block_img.size == 0:
                    row_maps.append(np.arange(level, dtype=np.uint8))
                    continue

                hists = self.calc_histogram_(block_img, level=level)
                clip_hists = self.clip_histogram_(hists, threshold=threshold)
                cdf = self.calc_histogram_cdf_(
                    clip_hists, block_img.shape[0], block_img.shape[1], level=level
                )
                row_maps.append(cdf)
            maps.append(row_maps)

        # 像素映射：此处采取“就近块映射”，稳定且实现简洁
        out = img_arr.copy().astype(np.float32)
        for i in range(m):
            for j in range(n):
                r = min(blocks - 1, int(i / block_m))
                c = min(blocks - 1, int(j / block_n))
                out[i, j] = maps[r][c][img_arr[i, j]]
        return out.astype(np.uint8)

    def bright_wise_histogram_equalization(self, img_arr, level=256, **kwargs):
        """
        按亮度分段均衡化

        思路:
        1) 通过累计直方图找到两个分割点(约 1/3、2/3 分位)
        2) 将图像分成暗/中/亮三段
        3) 每段分别做“段内均衡化”
        4) 三段结果拼回原图
        """
        hists = self.calc_histogram_(img_arr, level=level)
        cumsum = np.cumsum(np.array(hists))
        ratio = cumsum / cumsum[-1]

        scale1, scale2 = None, None
        for i, v in enumerate(ratio):
            if v >= 0.333 and scale1 is None:
                scale1 = i
            if v >= 0.667 and scale2 is None:
                scale2 = i
                break

        if scale1 is None:
            scale1 = int(level / 3)
        if scale2 is None:
            scale2 = int(2 * level / 3)

        dark_idx = img_arr <= scale1
        mid_idx = (img_arr > scale1) & (img_arr <= scale2)
        bright_idx = img_arr > scale2

        out = np.zeros_like(img_arr, dtype=np.uint8)

        # 暗区均衡
        if np.any(dark_idx):
            dark_h = self._special_histogram(img_arr[dark_idx], 0, scale1)
            dark_cdf = self._special_histogram_cdf(dark_h, 0, scale1)
            out[dark_idx] = dark_cdf[img_arr[dark_idx]]

        # 中间区均衡
        if np.any(mid_idx):
            mid_h = self._special_histogram(img_arr[mid_idx], scale1, scale2)
            mid_cdf = self._special_histogram_cdf(mid_h, scale1, scale2)
            out[mid_idx] = mid_cdf[img_arr[mid_idx] - scale1]

        # 亮区均衡
        if np.any(bright_idx):
            bright_h = self._special_histogram(img_arr[bright_idx], scale2, level - 1)
            bright_cdf = self._special_histogram_cdf(bright_h, scale2, level - 1)
            out[bright_idx] = bright_cdf[img_arr[bright_idx] - scale2]

        return out.astype(np.uint8)

    @staticmethod
    def standard_histogram_equalization(img_arr, level=256, **kwargs):
        """
        使用 PIL 提供的标准均衡化接口。
        """
        img = Image.fromarray(img_arr)
        return np.array(ImageOps.equalize(img)).astype(np.uint8)

    @staticmethod
    def calc_histogram_(gray_arr, level=256):
        """
        统计灰度直方图

        返回长度为 level 的 list，第 i 项表示灰度 i 的像素数。
        """
        hists = [0 for _ in range(level)]
        for row in gray_arr:
            for p in row:
                hists[int(p)] += 1
        return hists

    @staticmethod
    def calc_histogram_cdf_(hists, block_m, block_n, level=256):
        """
        根据直方图计算 CDF 映射表。

        公式:
            cdf[k] = (level - 1) / (block_m * block_n) * cumulative_hist[k]
        """
        hists_cumsum = np.cumsum(np.array(hists))
        const_a = (level - 1) / max(1, (block_m * block_n))
        return (const_a * hists_cumsum).astype(np.uint8)

    @staticmethod
    def clip_histogram_(hists, threshold=10.0):
        """
        直方图裁剪函数(用于 CLAHE)。

        做法:
        - 计算平均值，得到裁剪上限 threshold_value
        - 超过上限的部分记为“多余量”
        - 把多余量均匀分配回所有灰度级，防止局部尖峰过大
        """
        all_sum = sum(hists)
        threshold_value = all_sum / max(1, len(hists)) * threshold

        total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
        mean_extra = total_extra / max(1, len(hists))

        out = [0 for _ in hists]
        for i in range(len(hists)):
            if hists[i] >= threshold_value:
                out[i] = int(threshold_value + mean_extra)
            else:
                out[i] = int(hists[i] + mean_extra)
        return out

    @staticmethod
    def _special_histogram(values, min_v, max_v):
        """
        统计指定灰度区间 [min_v, max_v] 内的子直方图。
        """
        hists = [0 for _ in range(max_v - min_v + 1)]
        for v in values:
            hists[int(v) - min_v] += 1
        return hists

    @staticmethod
    def _special_histogram_cdf(hists, min_v, max_v):
        """
        计算指定灰度区间内的 CDF 映射，输出仍落在该区间内。
        """
        hists_cumsum = np.cumsum(np.array(hists))
        cdf = (max_v - min_v) / max(1, hists_cumsum[-1]) * hists_cumsum + min_v
        return cdf.astype(np.uint8)

