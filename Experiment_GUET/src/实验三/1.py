# 任务5（进阶）：实现4个不同阶段目标（含鼠标手工标注）
# 1) 鼠标点击手工去除对称频率
# 2) 用方框标注探测频率
# 3) 程序自动选择对称频率
# 4) 同时去除位于图像边缘的周期性噪声

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'Experiment_GUET').exists():
            return p
    raise FileNotFoundError('未找到包含 Experiment_GUET 的项目根目录')


PROJECT_ROOT = find_project_root()
PIC_DIR = PROJECT_ROOT / 'Experiment_GUET' / 'pic'
CLASS_IMG_DIR = PROJECT_ROOT / 'Experiment_GUET' / 'src' / 'class_img'
OUT_DIR = PROJECT_ROOT / 'Experiment_GUET' / 'src' / '实验三' / 'results_strict'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return img


def BWBand(w: int, h: int):
    img = np.zeros((321, 201), dtype=np.uint8)
    center_row = 321 // 2
    center_col = 201 // 2
    start_y = center_row - h // 2
    end_y = center_row + h // 2
    start_x = center_col - w // 2
    end_x = center_col + w // 2
    img[start_y:end_y, start_x:end_x] = 255
    return img


def fft_spectrum(gray):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log1p(mag)
    return f, fshift, mag, log_mag


def normalize_u8(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def show_and_save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches='tight')
    plt.show()


def rotate_image(img, angle):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def add_salt_pepper_noise(img, amount=0.03, salt_vs_pepper=0.5):
    noisy = img.copy()
    total = img.size
    num_salt = int(total * amount * salt_vs_pepper)
    num_pepper = int(total * amount * (1 - salt_vs_pepper))

    ys = np.random.randint(0, img.shape[0], num_salt)
    xs = np.random.randint(0, img.shape[1], num_salt)
    noisy[ys, xs] = 255

    ys = np.random.randint(0, img.shape[0], num_pepper)
    xs = np.random.randint(0, img.shape[1], num_pepper)
    noisy[ys, xs] = 0
    return noisy


def gaussian_lowpass_filter(shape, d0):
    m, n = shape
    y = np.arange(m, dtype=np.float64)[:, None] - m / 2.0
    x = np.arange(n, dtype=np.float64)[None, :] - n / 2.0
    d2 = x * x + y * y
    return np.exp(-d2 / (2.0 * (d0 ** 2)))


def apply_frequency_filter(img, h_filter):
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    gshift = fshift * h_filter
    g = np.fft.ifft2(np.fft.ifftshift(gshift))
    return normalize_u8(np.real(g)), normalize_u8(np.log1p(np.abs(gshift)))


def peak_points_from_spectrum(log_spec, center, k=8, center_radius=12):
    s = log_spec.copy()
    cy, cx = center
    yy, xx = np.indices(s.shape)
    mask_center = (yy - cy) ** 2 + (xx - cx) ** 2 <= center_radius ** 2
    s[mask_center] = -np.inf

    flat_idx = np.argpartition(s.ravel(), -k)[-k:]
    pts = [np.unravel_index(i, s.shape) for i in flat_idx]
    pts = sorted(pts, key=lambda p: s[p], reverse=True)
    return pts


def notch_reject(shape, points, radius=4):
    h, w = shape
    yy, xx = np.indices((h, w))
    mask = np.ones((h, w), dtype=np.float64)
    for y, x in points:
        rr = (yy - y) ** 2 + (xx - x) ** 2
        mask[rr <= radius * radius] = 0.0
    return mask


def split_peak_groups(points, center):
    cy, cx = center
    horizontal, vertical, diagonal = [], [], []
    for y, x in points:
        dy = y - cy
        dx = x - cx
        if abs(dy) <= max(1, 0.45 * abs(dx)):
            horizontal.append((y, x))
        elif abs(dx) <= max(1, 0.45 * abs(dy)):
            vertical.append((y, x))
        else:
            diagonal.append((y, x))
    return horizontal[:2], vertical[:2], diagonal[:4]


print(f'项目根目录: {PROJECT_ROOT}')
print(f'结果目录: {OUT_DIR}')
np.random.seed(0)



img = read_gray(CLASS_IMG_DIR / 'Fig0464(a)(car_75DPI_Moire).tif')
F = np.fft.fft2(img.astype(np.float64))
Fshift = np.fft.fftshift(F)
log_spec = np.log1p(np.abs(Fshift))
H, W = img.shape
cy, cx = H // 2, W // 2


def notch_mask(shape, points, radius=4):
    m, n = shape
    yy, xx = np.indices((m, n))
    mask = np.ones((m, n), dtype=np.float64)
    for y, x in points:
        mask[(yy - y) ** 2 + (xx - x) ** 2 <= radius * radius] = 0.0
    return mask


def apply_mask_and_reconstruct(fshift, mask):
    gshift = fshift * mask
    g = np.fft.ifft2(np.fft.ifftshift(gshift))
    out = normalize_u8(np.real(g))
    out_spec = normalize_u8(np.log1p(np.abs(gshift)))
    return out, out_spec


def symmetric_from_clicks(click_points, center):
    """把鼠标点击点自动补成关于频谱中心的对称点对。"""
    c0, c1 = center
    pts = []
    for x, y in click_points:
        x = int(round(x))
        y = int(round(y))
        pts.append((y, x))
        pts.append((2 * c0 - y, 2 * c1 - x))
    out = []
    for y, x in pts:
        if 0 <= y < H and 0 <= x < W and (y, x) not in out:
            out.append((y, x))
    return out


# -------------------- 1) 鼠标手工标注对称频率 --------------------
plt.figure(figsize=(7, 6))
plt.imshow(normalize_u8(log_spec), cmap='gray')
plt.title('Task1 click noise peaks (left click), middle/right click or Enter to finish')
plt.axis('on')
clicked = plt.ginput(n=-1, timeout=0)
plt.close()

# 若没有点击，给一组默认点，保证代码可继续运行
if len(clicked) == 0:
    clicked = [(cx + 28, cy + 38), (cx - 24, cy + 40)]

manual_pts = symmetric_from_clicks(clicked, (cy, cx))
manual_mask = notch_mask(img.shape, manual_pts, radius=4)
img_manual, spec_manual = apply_mask_and_reconstruct(Fshift, manual_mask)

# -------------------- 2) 方框标注探测频率 --------------------
box_half = 8
box_vis = normalize_u8(log_spec).copy()
box_vis = cv2.cvtColor(box_vis, cv2.COLOR_GRAY2BGR)
box_mask = np.ones(img.shape, dtype=np.float64)
for (y, x) in manual_pts:
    y0, y1 = max(0, y - box_half), min(H, y + box_half + 1)
    x0, x1 = max(0, x - box_half), min(W, x + box_half + 1)
    cv2.rectangle(box_vis, (x0, y0), (x1, y1), (255, 255, 255), 1)
    box_mask[y0:y1, x0:x1] = 0.0
img_box, spec_box = apply_mask_and_reconstruct(Fshift, box_mask)

# -------------------- 3) 自动选择对称频率 --------------------
auto_peaks = peak_points_from_spectrum(log_spec, center=(cy, cx), k=8, center_radius=12)
auto_mask = notch_mask(img.shape, auto_peaks, radius=4)
img_auto, spec_auto = apply_mask_and_reconstruct(Fshift, auto_mask)

# -------------------- 4) 同时去除边缘周期噪声 --------------------
edge_band = 18
cands = peak_points_from_spectrum(log_spec, center=(cy, cx), k=24, center_radius=12)
edge_pts = []
for y, x in cands:
    if y < edge_band or y >= H - edge_band or x < edge_band or x >= W - edge_band:
        edge_pts.append((y, x))
edge_pts = edge_pts[:8]
all_pts = list(dict.fromkeys(auto_peaks + edge_pts))
edge_mask = notch_mask(img.shape, all_pts, radius=4)
img_edge, spec_edge = apply_mask_and_reconstruct(Fshift, edge_mask)

# -------------------- 可视化 --------------------
fig, ax = plt.subplots(3, 5, figsize=(22, 12))

ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].set_title('Original image')
ax[1, 0].imshow(normalize_u8(log_spec), cmap='gray')
ax[1, 0].set_title('Original power spectrum')
ax[2, 0].axis('off')

ax[0, 1].imshow(img_manual, cmap='gray')
ax[0, 1].set_title('1) Manual click-notch result')
ax[1, 1].imshow(spec_manual, cmap='gray')
ax[1, 1].set_title('1) Spectrum after manual notch')
ax[2, 1].text(0.02, 0.5, f'manual_pts={manual_pts}', fontsize=9)
ax[2, 1].axis('off')

ax[0, 2].imshow(img_box, cmap='gray')
ax[0, 2].set_title('2) Box-mark suppression result')
ax[1, 2].imshow(box_vis[:, :, ::-1])
ax[1, 2].set_title('2) Box-marked spectrum')
ax[2, 2].imshow(spec_box, cmap='gray')
ax[2, 2].set_title('2) Spectrum after box suppression')

ax[0, 3].imshow(img_auto, cmap='gray')
ax[0, 3].set_title('3) Auto-peak notch result')
ax[1, 3].imshow(spec_auto, cmap='gray')
ax[1, 3].set_title('3) Spectrum after auto notch')
ax[2, 3].text(0.02, 0.5, f'auto_peaks={auto_peaks}', fontsize=9)
ax[2, 3].axis('off')

ax[0, 4].imshow(img_edge, cmap='gray')
ax[0, 4].set_title('4) Auto peaks + edge periodic noise')
ax[1, 4].imshow(spec_edge, cmap='gray')
ax[1, 4].set_title('4) Spectrum after combined notch')
ax[2, 4].text(0.02, 0.5, f'edge_pts={edge_pts}', fontsize=9)
ax[2, 4].axis('off')

for a in ax.ravel():
    if a.has_data():
        a.axis('off')

fig.tight_layout()
fig.savefig(OUT_DIR / 'task5_advanced_four_stage_targets.png', dpi=150, bbox_inches='tight')
plt.show()

print('clicked =', clicked)
print('manual_pts =', manual_pts)
print('auto_peaks =', auto_peaks)
print('edge_pts =', edge_pts)
print('Saved to:', OUT_DIR / 'task5_advanced_four_stage_targets.png')

