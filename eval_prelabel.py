#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_prelabel.py  （OCR 版）

比對「Gemini 全文 OCR 預標」與「人工修正後的正解」，算出三個指標：

  1. 偵測率 (detection)：有沒有框到該框的位置（只看框，IoU>=門檻即算命中，不管文字）。
  2. 辨識率 (recognition)：在框對的地方，文字有沒有讀對（NFKC 正規化後比對）。
  3. 綜合準確率 (end-to-end)：框對 + 文字對，這是最終的 90% 目標。

用法：
    python eval_prelabel.py --pred <預標資料夾> --gt <正解資料夾> [--iou 0.5] [--show 15]

兩邊用「同檔名」的 .xml 配對。
"""
import argparse
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pascal_voc_io import PascalVocReader  # noqa: E402


def norm_text(s):
    """文字正規化：全形轉半形、去空白、去千分位逗號、統一圈圈零(○↔〇)。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("○", "〇")          # 白圈 vs 表意零，視為同字
    s = s.replace(",", "").replace("，", "")  # 金額千分位不計較
    s = "".join(s.split())
    s = s.strip("-－—、.。:：")          # 去掉頭尾雜點（例如 800- → 800），保留內部 1-2
    return s


def load_boxes(xml_path):
    """回傳 [(text, xmin, ymin, xmax, ymax), ...]（旋轉框取外接正框）。"""
    reader = PascalVocReader(xml_path)
    boxes = []
    for shape in reader.getShapes():
        text = shape[0]
        pts = shape[1]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        boxes.append((text, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
    bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(pred, gt, thr):
    """只用 IoU 貪婪配對（不管文字）。回傳配對清單 [(pi, gi, iou)]。"""
    cand = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            v = iou(p, g)
            if v >= thr:
                cand.append((v, pi, gi))
    cand.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for v, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, v))
    return pairs, used_p, used_g


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description="計算 Gemini 全文 OCR 預標的準確率")
    ap.add_argument("--pred", required=True, help="預標 XML 資料夾")
    ap.add_argument("--gt", required=True, help="人工修正後正解 XML 資料夾")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU 門檻（預設 0.5）")
    ap.add_argument("--target", type=float, default=0.90, help="目標綜合準確率（預設 0.90）")
    ap.add_argument("--show", type=int, default=12, help="列出幾個錯誤範例（預設 12）")
    args = ap.parse_args()

    gt_files = sorted(f for f in os.listdir(args.gt) if f.endswith(".xml"))
    if not gt_files:
        sys.exit("正解資料夾裡沒有 .xml：%s" % args.gt)

    det_tp = det_fp = det_fn = 0        # 偵測
    reco_ok = reco_total = 0            # 辨識（在配對成功的框裡）
    e2e_tp = 0                          # 綜合正確
    wrong_text = []                     # (檔, pred文字, gt文字)
    missed = []                         # (檔, gt文字)  沒被框到
    spurious = []                       # (檔, pred文字) 多框的

    print("IoU 門檻：%.2f\n" % args.iou)
    print("%-26s %6s %6s %6s %6s" % ("檔名", "框對", "多框", "漏框", "字對"))
    print("-" * 54)
    for fname in gt_files:
        gt = load_boxes(os.path.join(args.gt, fname))
        pp = os.path.join(args.pred, fname)
        pred = load_boxes(pp) if os.path.exists(pp) else []

        pairs, used_p, used_g = match(pred, gt, args.iou)
        d_tp = len(pairs)
        d_fp = len(pred) - len(used_p)
        d_fn = len(gt) - len(used_g)

        r_ok = 0
        for pi, gi, _ in pairs:
            if norm_text(pred[pi][0]) == norm_text(gt[gi][0]):
                r_ok += 1
            else:
                wrong_text.append((fname, pred[pi][0], gt[gi][0]))
        for pi, p in enumerate(pred):
            if pi not in used_p:
                spurious.append((fname, p[0]))
        for gi, g in enumerate(gt):
            if gi not in used_g:
                missed.append((fname, g[0]))

        det_tp += d_tp
        det_fp += d_fp
        det_fn += d_fn
        reco_ok += r_ok
        reco_total += d_tp
        e2e_tp += r_ok

        note = "" if os.path.exists(pp) else " (無預標檔)"
        print("%-26s %6d %6d %6d %6d%s" % (fname[:26], d_tp, d_fp, d_fn, r_ok, note))

    dp, dr, df = prf(det_tp, det_fp, det_fn)
    reco_acc = reco_ok / reco_total if reco_total else 0.0
    # 綜合：框對且字對才算 TP；其餘 pred 為 FP、其餘 gt 為 FN
    e2e_fp = (det_tp - e2e_tp) + det_fp
    e2e_fn = (det_tp - e2e_tp) + det_fn
    ep, er, ef = prf(e2e_tp, e2e_fp, e2e_fn)

    print("\n===== 指標 =====")
    print("1) 偵測率  precision %.1f%% / recall %.1f%% / F1 %.1f%%" % (dp * 100, dr * 100, df * 100))
    print("   （框對 %d，多框 %d，漏框 %d）" % (det_tp, det_fp, det_fn))
    print("2) 辨識率  %.1f%%（框對的 %d 個裡，文字讀對 %d 個）" %
          (reco_acc * 100, reco_total, reco_ok))
    print("3) 綜合準確率（框對+字對）F1 %.1f%%  (precision %.1f%% / recall %.1f%%)" %
          (ef * 100, ep * 100, er * 100))
    print("-" * 40)
    if ef >= args.target:
        print("✅ 已達標！綜合 F1 %.1f%% >= 目標 %.0f%%" % (ef * 100, args.target * 100))
    else:
        print("⏳ 未達標：綜合 F1 %.1f%%，還差 %.1f 個百分點到 %.0f%%" %
              (ef * 100, (args.target - ef) * 100, args.target * 100))

    if args.show:
        if missed:
            print("\n--- 漏框（該框卻沒框到）前 %d 個 ---" % args.show)
            for fn, t in missed[:args.show]:
                print("   [%s] 缺：%r" % (fn, t))
        if wrong_text:
            print("\n--- 框到了但文字讀錯 前 %d 個（pred → gt）---" % args.show)
            for fn, pt, gt_ in wrong_text[:args.show]:
                print("   [%s] %r → 應為 %r" % (fn, pt, gt_))
        if spurious:
            print("\n--- 多框（框了正解沒有的）前 %d 個 ---" % args.show)
            for fn, t in spurious[:args.show]:
                print("   [%s] 多：%r" % (fn, t))


if __name__ == "__main__":
    main()
