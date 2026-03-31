"""
直方图均衡化示例：读入 pic 下图像，HE 后写回同一 pic 目录。
"""
from pathlib import Path

from PIL import Image

from Histogram_process import HistogramProcessor

# 与仓库内路径一致：Chapter03/src/pic/
_PIC_DIR = Path(__file__).resolve().parent.parent / "pic"
_INPUT_NAME = "langlanglangliang_genshin_Venti_daf5bd60-abaf-4bcf-9e1b-d017a60aac0d.png"


def main():
    in_path = _PIC_DIR / _INPUT_NAME
    if not in_path.is_file():
        raise FileNotFoundError(f"找不到输入图像: {in_path}")

    img = Image.open(in_path)
    processor = HistogramProcessor()
    out_img = processor.enhance_contrast(img, method="HE")

    out_path = _PIC_DIR / f"{in_path.stem}_he.png"
    out_img.save(out_path)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
