# DIP 项目说明

## 目录导航（建议先看）

- [快速开始](#快速开始)
- [知识点速查](#知识点速查)
  - [灰度变换](#灰度变换)
  - [直方图均衡化](#直方图均衡化)
  - [空间滤波](#空间滤波)
  - [拉普拉斯锐化滤波（原理+手算）](#拉普拉斯锐化滤波原理手算)
- [实验结果速查](#实验结果速查)
  - [实验 A：灰度变换与均衡化](#实验-a灰度变换与均衡化)
  - [实验 B：实验内容实现（四题）](#实验-b实验内容实现四题)

---

## 快速开始

### 1) 搭建环境

#### 推荐：Miniconda（下载地址）

[https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

```bash
conda create -n DIP python=3.8
conda activate DIP
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 或直接安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2) 运行入口

```bash
cd ./Experiment_GUET/src
python Grayscale_transformation.py
```

### 3) 结果文件位置

- `Experiment_GUET/src/image/`
- `Experiment_GUET/src/image/exp_content_results/`（新实验 notebook 结果）

---

## 知识点速查

## 灰度变换

灰度图像每个像素只有一个亮度值，常见取值范围为 `0~255`（`0` 黑，`255` 白）。

### 1) 图像反转

适合突出"原图中偏暗背景上的亮目标"。

```
T(r) = L - r
```

### 2) 对数变换

![](./Experiment_GUET/pic/对数变换.png)

低灰度区拉伸，高灰度区压缩，常用于增强暗部细节。

```
s = c * log(1 + r)
```

### 3) 幂律（伽马）变换

![](./Experiment_GUET/pic/伽马变换.png)

- `gamma < 1`：图像整体提亮
- `gamma > 1`：图像整体压暗并增强亮区对比

```
s = c * r^gamma
```

---

## 直方图均衡化

流程（离散 256 级）：

1. 统计每个灰度级出现概率（PDF）
2. 计算累计分布函数（CDF）
3. 按映射公式重映射灰度

```
s_k = round((L - 1) * cdf[k])   (L = 256)
```

作用：把灰度分布拉开，通常可提升整体对比度。

---

## 空间滤波

空间滤波是在图像平面上，对"像素邻域"做运算得到新像素值。

### 1) 线性滤波（卷积）

- 典型：均值、高斯、Sobel、Prewitt
- 本质：邻域像素与核系数逐项乘加

### 2) 非线性滤波

- 典型：中值、最大值、最小值
- 本质：邻域排序或取极值，不是卷积乘加

### 3) 边界处理

邻域越界时需要 padding。不同 padding 仅影响图像外圈（厚度约核半径）。

- OpenCV 默认常接近反射策略（`BORDER_DEFAULT`）
- 手写实现常见常数 0 填充（`BORDER_CONSTANT`）

### 4) 常见效果

- 平滑类：降噪，但会模糊边缘
- 梯度/拉普拉斯：增强边缘，但可能放大噪声

---

## 拉普拉斯锐化滤波（原理+手算）

拉普拉斯是二阶导，平坦区响应接近 0，突变区响应大。

$$
\nabla^2 f=\frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 f}{\partial y^2}
$$

常见离散核（3×3）：

$$
\begin{bmatrix}
0&-1&0\\
-1&4&-1\\
0&-1&0
\end{bmatrix}
\quad \text{或} \quad
\begin{bmatrix}
-1&-1&-1\\
-1&8&-1\\
-1&-1&-1
\end{bmatrix}
$$

锐化叠加常写为：

$$
g=f-\alpha L
$$

（若模板符号相反，也可写成 g = f + αL，关键是前后一致）

### 手算示例

邻域灰度块：

$$
\begin{bmatrix}
98 & 102 & 100 \\
97 & 100 & 103 \\
96 & 99 & 101
\end{bmatrix}
$$

取 4 邻域核，中心像素 fc=100，上/下/左/右为 102,99,97,103：

$$
L = 4f_c-(f_{up}+f_{down}+f_{left}+f_{right})=4\times100-(102+99+97+103)=-1
$$

取 α=1，锐化后：

$$
g=100-(-1)=101
$$

最后要裁剪到合法灰度范围：

$$
g\leftarrow \text{clip}(g,0,255)
$$

---

## 实验结果速查

## 实验 A：灰度变换与均衡化

对应文件：

- `Experiment_GUET/src/Grayscale_transformation.py`
- `Experiment_GUET/src/Grayscale_transformation.ipynb`

典型输出：

- `Venti_gray.jpg`
- `Venti_reversed.jpg`
- `Venti_log.jpg`
- `Venti_gamma.jpg`
- `Venti_histogram_equalization.jpg`

结论（简要）：

- 反转可突出暗背景下亮目标
- 对数变换有利于暗部细节提升
- 伽马变换用于全局亮度/对比控制
- 直方图均衡化可扩展灰度动态范围

---

## 实验 B：实验内容实现（四题）

对应文件：

- `Experiment_GUET/src/Experiment_Content_Implementation.ipynb`

### B1 两幅 64×64 图的直方图比较 + 滤波后比较

- 模板（3×3 高斯核）：
  $$
  W=\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}
  $$
  边界 0 填充
- 结果：**滤波前直方图一致，滤波后不一致**
- 解释：卷积引入空间结构信息，空间分布不同会导致滤波后统计不同

### B2 三类噪声与滤波对比（椒盐/高斯/斑点）

- **椒盐噪声**：中值滤波最佳，脉冲噪点抑制明显
- **高斯噪声**：均值滤波更稳定，但细节有损失
- **斑点噪声**：平滑可缓解，但纹理细节会被削弱

### B3 Sobel 两方向响应

- 水平边缘核：
  $$
  \begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}
  $$
- 垂直边缘核：
  $$
  \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}
  $$
- 梯度幅值图可获得更完整边缘

### B4 隐形眼镜边界提取

流程：灰度化（输入若为彩色）→ 高斯平滑 → Canny → 闭运算 → 轮廓评分筛选

结果文件：

- `lens_boundary_only.png`
- `lens_overlay.png`

调参建议：边界断裂时优先调整 Canny 双阈值与闭运算核大小。
