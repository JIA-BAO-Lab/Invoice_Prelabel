#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preview_boxes.py

把預標 XML 的框（含中文 label）疊到原圖上，輸出成預覽 PNG，方便『沒有正解』時目視檢查。

用法：
    python preview_boxes.py <圖片資料夾> <預標XML資料夾> [輸出資料夾]

- 圖片資料夾：原始發票圖。
- 預標XML資料夾：gemini_prelabel.py 產生的某個 gemini_*/ 資料夾。
- 輸出資料夾：預設為「預標XML資料夾/preview」。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pascal_voc_io import PascalVocReader  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# macOS 常見 CJK 字型
CJK_FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def load_font(size):
    for p in CJK_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def main():
    if len(sys.argv) < 3:
        sys.exit("用法：python preview_boxes.py <圖片資料夾> <預標XML資料夾> [輸出資料夾]")
    img_dir, pred_dir = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(pred_dir, "preview")
    os.makedirs(out_dir, exist_ok=True)
    font = load_font(26)

    imgs = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(IMAGE_EXTS)]
    n = 0
    for f in imgs:
        stem = os.path.splitext(f)[0]
        xml = os.path.join(pred_dir, stem + ".xml")
        if not os.path.exists(xml):
            continue
        im = Image.open(os.path.join(img_dir, f)).convert("RGB")
        dr = ImageDraw.Draw(im)
        for s in PascalVocReader(xml).getShapes():
            label = s[0]
            xs = [p[0] for p in s[1]]
            ys = [p[1] for p in s[1]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            # 標籤底色 + 文字
            tw = len(label) * 16 + 6
            dr.rectangle([x1, max(0, y1 - 30), x1 + tw, y1], fill=(255, 0, 0))
            dr.text((x1 + 3, max(0, y1 - 29)), label, fill=(255, 255, 255), font=font)
        im.thumbnail((1500, 1500))
        out = os.path.join(out_dir, "pv_" + stem + ".png")
        im.save(out)
        print("→ %s（%d 個框）" % (out, len(PascalVocReader(xml).getShapes())))
        n += 1
    print("\n完成，共 %d 張預覽，存於：%s" % (n, out_dir))


if __name__ == "__main__":
    main()
