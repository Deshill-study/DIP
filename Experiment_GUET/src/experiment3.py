import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def fft2_image(image):
    """对图像进行二维傅里叶变换，并进行频谱中心化"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.float64(image)
    F = np.fft.fft2(f)
    Fshift = np.fft.fftshift(F)
    return Fshift


def ifft2_image(Fshift):
    """逆傅里叶变换，返回用于显示的空域 uint8（min-max 拉伸，高通时避免负值被 clip 掉）。"""
    F = np.fft.ifftshift(Fshift)
    f = np.fft.ifft2(F)
    img = np.real(f)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img)


def get_distance_matrix(M, N):
    """D(u,v)：与 fftshift 对齐，为各频率点到中心 (M/2, N/2) 的欧氏距离（教材常用定义）。"""
    row = np.arange(M, dtype=np.float64)[:, None] - M / 2.0
    col = np.arange(N, dtype=np.float64)[None, :] - N / 2.0
    return np.sqrt(row**2 + col**2)


def ideal_lowpass(M, N, D0):
    """理想低通滤波器"""
    D = get_distance_matrix(M, N)
    H = np.double(D <= D0)
    return H


def ideal_highpass(M, N, D0):
    """理想高通滤波器"""
    D = get_distance_matrix(M, N)
    H = np.double(D > D0)
    return H


def butterworth_lowpass(M, N, D0, n=2):
    """巴特沃斯低通滤波器"""
    D = get_distance_matrix(M, N)
    H = 1 / (1 + (D / D0)**(2*n))
    return H


def butterworth_highpass(M, N, D0, n=2):
    """巴特沃斯高通滤波器"""
    D = get_distance_matrix(M, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = 1 / (1 + (D0 / D) ** (2 * n))
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    H[D == 0] = 0.0
    return H


def gaussian_lowpass(M, N, D0):
    """高斯低通滤波器"""
    D = get_distance_matrix(M, N)
    H = np.exp(-(D**2) / (2 * D0**2))
    return H


def gaussian_highpass(M, N, D0):
    """高斯高通滤波器"""
    D = get_distance_matrix(M, N)
    H = 1 - np.exp(-(D**2) / (2 * D0**2))
    return H


def display_spectrum(Fshift):
    """显示频谱（对数变换增强可视化）"""
    magnitude = np.abs(Fshift)
    magnitude_log = np.log(1 + magnitude)
    magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude_norm)


def main():
    """主函数：运行所有实验任务"""
    
    # 测试图片路径（请修改为你的图片路径）
    image_path = '/home/ubuntu/codebase/yexijia/github_project/DIP/Experiment_GUET/src/image/Venti_gray.jpg'
    
    print("=" * 60)
    print("频域图像增强实验")
    print("=" * 60)
    
    # 检查图片是否存在
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        print("请确保图片文件存在，或修改 image_path 变量")
        return
    
    print(f"成功读取图像：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M, N = gray.shape
    print(f"图像尺寸：{M} x {N}")
    
    # 傅里叶变换
    print("\n正在进行傅里叶变换...")
    Fshift = fft2_image(gray)
    
    # 任务1：显示原图和频谱
    print("\n任务1：生成原图和频谱...")
    spectrum = display_spectrum(Fshift)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(spectrum, cmap='gray')
    plt.title('Fourier Spectrum (Log)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('01_original_and_spectrum.png', dpi=150, bbox_inches='tight')
    print("保存：01_original_and_spectrum.png")
    plt.close()
    
    # 任务2-7：各种滤波器
    D0_values = [30, 60, 120]
    n_values = [1, 2, 4]
    
    filters = [
        ('理想低通', ideal_lowpass, '02_ideal_lowpass'),
        ('理想高通', ideal_highpass, '03_ideal_highpass'),
        ('巴特沃斯低通', lambda m,n,d: butterworth_lowpass(m,n,d,2), '04_butterworth_lowpass'),
        ('巴特沃斯高通', lambda m,n,d: butterworth_highpass(m,n,d,2), '05_butterworth_highpass'),
        ('高斯低通', gaussian_lowpass, '06_gaussian_lowpass'),
        ('高斯高通', gaussian_highpass, '07_gaussian_highpass'),
    ]
    
    for name, filter_func, filename in filters:
        print(f"\n生成：{name}滤波器...")
        
        plt.figure(figsize=(15, 10))
        for i, D0 in enumerate(D0_values):
            H = filter_func(M, N, D0)
            G = Fshift * H
            result = ifft2_image(G)
            
            plt.subplot(3, 3, i*3 + 1)
            plt.imshow(H, cmap='gray')
            plt.title(f'Filter D0={D0}')
            plt.axis('off')
            
            plt.subplot(3, 3, i*3 + 2)
            plt.imshow(np.log(1 + np.abs(G)), cmap='gray')
            plt.title(f'Spectrum D0={D0}')
            plt.axis('off')
            
            plt.subplot(3, 3, i*3 + 3)
            plt.imshow(result, cmap='gray')
            plt.title(f'Result D0={D0}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
        print(f"保存：{filename}.png")
        plt.close()
    
    # 任务8：巴特沃斯不同阶数对比
    print("\n任务8：巴特沃斯不同阶数对比...")
    D0 = 60
    
    plt.figure(figsize=(15, 10))
    for i, n in enumerate(n_values):
        # 低通
        H_lp = butterworth_lowpass(M, N, D0, n)
        G_lp = Fshift * H_lp
        result_lp = ifft2_image(G_lp)
        
        # 高通
        H_hp = butterworth_highpass(M, N, D0, n)
        G_hp = Fshift * H_hp
        result_hp = ifft2_image(G_hp)
        
        plt.subplot(3, 4, i*4 + 1)
        plt.imshow(H_lp, cmap='gray')
        plt.title(f'BLPF n={n}')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 2)
        plt.imshow(result_lp, cmap='gray')
        plt.title(f'Lowpass n={n}')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 3)
        plt.imshow(H_hp, cmap='gray')
        plt.title(f'BHPF n={n}')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 4)
        plt.imshow(result_hp, cmap='gray')
        plt.title(f'Highpass n={n}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('08_butterworth_order_comparison.png', dpi=150, bbox_inches='tight')
    print("保存：08_butterworth_order_comparison.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("所有实验完成！生成的图片文件：")
    print("=" * 60)
    print("01_original_and_spectrum.png - 原图和频谱")
    print("02_ideal_lowpass.png - 理想低通滤波器")
    print("03_ideal_highpass.png - 理想高通滤波器")
    print("04_butterworth_lowpass.png - 巴特沃斯低通滤波器")
    print("05_butterworth_highpass.png - 巴特沃斯高通滤波器")
    print("06_gaussian_lowpass.png - 高斯低通滤波器")
    print("07_gaussian_highpass.png - 高斯高通滤波器")
    print("08_butterworth_order_comparison.png - 巴特沃斯阶数对比")
    print("=" * 60)


if __name__ == '__main__':
    main()