#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_prelabel.py

用 Google Gemini 對發票影像做「自動預標」，每張圖旁邊產生同名的 PascalVOC
XML（roLabelImg 直接可開），之後在 roLabelImg 裡人工修正即可。

用法：
    export GEMINI_API_KEY=你的金鑰
    python gemini_prelabel.py <圖片檔或資料夾> [選項]

範例：
    python gemini_prelabel.py ./invoices
    python gemini_prelabel.py ./invoices --classes data/invoice_classes.txt --overwrite

金鑰：從環境變數 GEMINI_API_KEY 或 GOOGLE_API_KEY 讀取，程式本身不儲存金鑰。
"""
import argparse
import json
import os
import sys
import time

from PIL import Image

# 重用 roLabelImg 自己的 XML 寫入器，確保格式 100% 相容
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pascal_voc_io import PascalVocWriter  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MODEL = "gemini-3.8-flash"


def build_prompt(classes=None):
    return (
        "你是一個台灣統一發票的 OCR 標注助手。請把這張發票影像上『特定的文字元素』"
        "一塊一塊框出來，並讀出每一塊的文字內容。\n\n"
        "【要框的（present 才框）】：\n"
        "- 發票號碼：開頭的字軌英文字母（例如 ZX、ZZ、DA）一個框，後面的數字另一個框。\n"
        "- 年月份那行（例如『一一五年五、六月份』）。\n"
        "- 開立日期整串（例如『中華民國115年5月31日』）。\n"
        "- 買受人的統一編號數字（通常印在一格一格的格子裡）。\n"
        "- 賣方（印章內）的統一編號數字，以及『統一編號』『統一編號:』這些標題字。\n"
        "- 表頭欄位字（逐字各一框）：銷、售、額、合、計、營、業、稅、課、別、總 等。\n"
        "- 『應稅』『零稅率』『免稅』：各自框成『一個詞一個框』，不要拆成單字。\n"
        "- 打勾符號 ✓。\n"
        "- 金額數字（銷售額、稅額、總計等），以及大寫金額整串"
        "（例如『億仟佰參拾肆萬肆仟陸佰壹拾元』要整串一個框，不要只框其中幾個字）。\n\n"
        "【不要框的】：公司名稱、買受人名稱、地址、電話、品項名稱、單價數量等自由文字，"
        "一律不要框。印章上的說明字『營業人蓋用統一發票專用章』『統一發票專用章』"
        "『專用章』這類都不要框（印章裡只框『統一編號』這幾個字以及統編數字）。\n\n"
        "輸出格式：\n"
        "- 只回傳一個 JSON 陣列，不要有任何多餘文字或 markdown。\n"
        '- 每個元素：{"text": "<讀到的文字>", "box_2d": [ymin, xmin, ymax, xmax]}\n'
        "- 座標一律正規化到 0 到 1000 的整數（相對於整張圖的比例）。\n\n"
        "切塊粒度規則（判斷依據是『版面上字有沒有分開』）：\n"
        "1. 字與字明顯分開（印在一格格裡、或字距很大）就『逐字各一個框』，text 放單一字元。"
        "例如：買受人統一編號那排數字、表頭『銷 售 額 合 計』『營 業 稅』『總 計』『課 稅 別』。\n"
        "2. 字連續排在一起就框成『一整塊』，text 放完整內容。"
        "例如：發票號碼 8 位數字、印章內連續的統編、日期整串、各個金額數字、大寫金額。\n"
        "3. text 要照實填寫讀到的字；讀不確定也盡量填最接近的。\n"
        "4. 框要緊貼文字，不要框到空白格線。\n"
        "5. 金額數字不要加千分位逗號（寫 328200 不要寫 328,200）；看清楚位數，不要多讀或少讀 0。\n"
        "6. 印章裡的統一編號數字是連續的，要框成『一整串一個框』，不要逐字拆開。\n"
        "7. 每個文字元素只框『一次』，不要重複框同一個東西。\n\n"
        "特別容易漏掉、請務必框出（present 才框）：\n"
        "- 買受人的統一編號：通常是 8 個數字、一格一個，請『逐一框出全部 8 個數字』（各一個框）。\n"
        "- 『統一編號』『統一編號:』這些標題字本身也要各框一個。\n"
        "- 稅別欄（應稅／零稅率／免稅）附近通常有一個手寫打勾，請框出來，text 用 \"✓\"。\n"
        "- 表頭每個字（銷/售/額/合/計、營/業/稅、課/稅/別、總/計）即使很小也要逐一框出，不要漏。\n"
        "- 年月份那行、開立日期、大寫金額整串，都不要漏。\n"
        "- 買方『統一編號:』（含冒號）這個標題字要框；右上角的聯式標記"
        "（例如『1-2』）若有也要框。\n\n"
        "定位精準度要求（非常重要）：\n"
        "- 這是一張『寬螢幕橫式』發票，寬度明顯大於高度；請正確判斷每個字的垂直位置，"
        "不要把下半部的東西畫得偏上。\n"
        "- 每個框都要『緊緊包住』該文字，上下左右的邊界都貼齊字的邊緣，邊距越小越好。\n"
        "- 逐字框（單一數字或單一表頭字）要特別精準，框大小約等於那一個字本身。\n"
        "- 回傳前請再次檢查每個框的 ymin/ymax 是否真的對應到該字在圖片中的實際高度位置。\n"
    )


def extract_json_array(text):
    """從模型回覆中盡量取出 JSON 陣列。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 圍籬
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 搶救 1：結尾截斷 → 取到最後一個完整 '}' 再補 ']'
    s = text[text.find("["):] if "[" in text else text
    last = s.rfind("}")
    if last != -1:
        try:
            return json.loads(s[:last + 1] + "]")
        except json.JSONDecodeError:
            pass
    # 搶救 2：中間有壞格式 → 用正規表達式逐一撈出合法的物件（容忍物件間的錯誤）
    return harvest_objects(text)


def harvest_objects(text):
    """從壞掉的 JSON 文字中，逐一撈出 {text, box_2d} 物件（兩種鍵順序都支援）。"""
    import re
    box = r'"box_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    txt = r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"'
    pat_a = re.compile(txt + r'\s*,\s*' + box, re.S)          # text 在前
    pat_b = re.compile(box + r'\s*,\s*' + txt, re.S)          # box 在前
    out = []
    for m in pat_a.finditer(text):
        t = json.loads('"' + m.group(1) + '"')               # 還原跳脫字元
        out.append({"text": t, "box_2d": [int(m.group(i)) for i in range(2, 6)]})
    for m in pat_b.finditer(text):
        t = json.loads('"' + m.group(5) + '"')
        out.append({"text": t, "box_2d": [int(m.group(i)) for i in range(1, 5)]})
    return out


def detect(client, model, image_path, prompt, retries=3):
    from google.genai import types

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext in ("tif", "tiff"):
        mime = "image/tiff"
    else:
        mime = "image/" + ext

    resp = None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                ),
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                print("    （%s 第 %d 次失敗，重試中…）" % (os.path.basename(image_path), attempt))
                time.sleep(2 * attempt)
    if resp is None:
        raise last_err
    u = getattr(resp, "usage_metadata", None)
    tok = {
        "in": getattr(u, "prompt_token_count", 0) or 0,
        "out": getattr(u, "candidates_token_count", 0) or 0,
        "total": getattr(u, "total_token_count", 0) or 0,
    }
    return extract_json_array(resp.text), tok


def to_pixels(box_2d, width, height):
    """box_2d = [ymin, xmin, ymax, xmax]，0-1000 正規化 -> 實際像素 (xmin,ymin,xmax,ymax)。"""
    ymin, xmin, ymax, xmax = box_2d
    x1 = int(round(xmin / 1000.0 * width))
    y1 = int(round(ymin / 1000.0 * height))
    x2 = int(round(xmax / 1000.0 * width))
    y2 = int(round(ymax / 1000.0 * height))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # 夾在圖片範圍內，並確保至少 1px
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def write_xml(image_path, detections, width, height, depth, out_dir=None):
    folder = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    filename = os.path.basename(image_path)
    writer = PascalVocWriter(folder, filename, (height, width, depth),
                             localImgPath=os.path.abspath(image_path))
    kept = 0
    for det in detections:
        # OCR 模式 label 就是讀到的文字；相容舊的 "label" 欄位
        text = det.get("text", det.get("label"))
        box = det.get("box_2d")
        if not text or not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = to_pixels(box, width, height)
        # 依需求輸出旋轉框 robndbox（Gemini 回正框，故角度為 0）
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        writer.addRotatedBndBox(cx, cy, w, h, 0.0, str(text), 0)
        kept += 1
    base = os.path.splitext(os.path.basename(image_path))[0] + ".xml"
    if out_dir:
        xml_path = os.path.join(out_dir, base)
    else:
        xml_path = os.path.splitext(image_path)[0] + ".xml"
    writer.save(targetFile=xml_path)
    return xml_path, kept


def gather_images(target):
    if os.path.isfile(target):
        return [target] if target.lower().endswith(IMAGE_EXTS) else []
    imgs = []
    for name in sorted(os.listdir(target)):
        if name.lower().endswith(IMAGE_EXTS):
            imgs.append(os.path.join(target, name))
    return imgs


def write_timing_only_report(out_dir, results, timing, model):
    """無正解時：只寫計時 + token + 每張框數，不評分。回傳 stats（scored=False）。"""
    import datetime
    stats = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model, "scored": False, "timing": timing,
        "per_image": [{"name": r["name"], "seconds": r["seconds"], "boxes": r["boxes"],
                       "tokens_in": r.get("tokens_in", 0), "tokens_out": r.get("tokens_out", 0)}
                      for r in results],
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    L = []
    L.append("發票 OCR 預標報告（無正解，未評分）")
    L.append("產生時間：%s" % stats["generated_at"])
    L.append("模型：%s" % model)
    L.append("")
    L.append("===== 計時 =====")
    L.append("處理張數：%d" % timing["processed"])
    L.append("每張平均：%.1f 秒（最快 %.1f，最慢 %.1f）" %
             (timing["avg_seconds"], timing["min_seconds"], timing["max_seconds"]))
    L.append("總花費：%.1f 秒（%.1f 分）" % (timing["total_seconds"], timing["total_seconds"] / 60.0))
    L.append("Token：輸入 %d，輸出 %d，總計 %d（每張平均 入%d/出%d）" % (
        timing.get("tokens_in", 0), timing.get("tokens_out", 0), timing.get("tokens_total", 0),
        timing.get("avg_tokens_in", 0), timing.get("avg_tokens_out", 0)))
    L.append("")
    L.append("===== 每張（框數）=====")
    L.append("%-30s %7s %6s %7s %7s" % ("檔名", "秒", "框數", "token入", "token出"))
    for pi in stats["per_image"]:
        L.append("%-30s %7.1f %6d %7d %7d" % (pi["name"][:30], pi["seconds"], pi["boxes"],
                                              pi["tokens_in"], pi["tokens_out"]))
    L.append("")
    L.append("（此批沒有正解 XML，故未計算準確率。請用疊圖或在 roLabelImg 開啟目視檢查。）")
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n----- 無正解，未評分（僅計時/token）-----")
    return stats


def write_reports(out_dir, gt_dir, results, timing, model, iou_thr=0.5, target=0.90):
    """比對 out_dir 的預標與 gt_dir 最上層的同名正解，把統計寫進 out_dir。"""
    import datetime
    # 重用 eval 的核心函式
    from eval_prelabel import load_boxes, match, prf, norm_text

    det_tp = det_fp = det_fn = 0
    reco_ok = reco_total = e2e_tp = 0
    per_image = []
    missed, wrong_text, spurious = [], [], []

    for r in results:
        name = r["name"]
        stem = os.path.splitext(name)[0]
        gt_path = os.path.join(gt_dir, stem + ".xml")
        pred_path = r["pred_xml"]
        if not os.path.exists(gt_path):
            per_image.append({"name": name, "seconds": r["seconds"], "boxes": r["boxes"],
                              "tokens_in": r.get("tokens_in", 0), "tokens_out": r.get("tokens_out", 0),
                              "gt_available": False})
            continue
        gt = load_boxes(gt_path)
        pred = load_boxes(pred_path) if os.path.exists(pred_path) else []
        pairs, used_p, used_g = match(pred, gt, iou_thr)
        d_tp = len(pairs)
        d_fp = len(pred) - len(used_p)
        d_fn = len(gt) - len(used_g)
        r_ok = 0
        for pi, gi, _ in pairs:
            if norm_text(pred[pi][0]) == norm_text(gt[gi][0]):
                r_ok += 1
            else:
                wrong_text.append((name, pred[pi][0], gt[gi][0]))
        for pi, p in enumerate(pred):
            if pi not in used_p:
                spurious.append((name, p[0]))
        for gi, g in enumerate(gt):
            if gi not in used_g:
                missed.append((name, g[0]))
        det_tp += d_tp; det_fp += d_fp; det_fn += d_fn
        reco_ok += r_ok; reco_total += d_tp; e2e_tp += r_ok
        per_image.append({"name": name, "seconds": r["seconds"], "boxes": r["boxes"],
                          "tokens_in": r.get("tokens_in", 0), "tokens_out": r.get("tokens_out", 0),
                          "gt_available": True, "gt_boxes": len(gt),
                          "hit": d_tp, "spurious": d_fp, "missed": d_fn, "text_ok": r_ok})

    dp, dr, df = prf(det_tp, det_fp, det_fn)
    reco_acc = reco_ok / reco_total if reco_total else 0.0
    e2e_fp = (det_tp - e2e_tp) + det_fp
    e2e_fn = (det_tp - e2e_tp) + det_fn
    ep, er, ef = prf(e2e_tp, e2e_fp, e2e_fn)

    stats = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "iou_threshold": iou_thr,
        "target_f1": target,
        "timing": timing,
        "per_image": per_image,
        "metrics": {
            "detection": {"precision": round(dp, 4), "recall": round(dr, 4), "f1": round(df, 4),
                          "tp": det_tp, "fp": det_fp, "fn": det_fn},
            "recognition_accuracy": round(reco_acc, 4),
            "end_to_end": {"precision": round(ep, 4), "recall": round(er, 4), "f1": round(ef, 4),
                           "tp": e2e_tp},
            "reached_target": bool(ef >= target),
        },
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("發票 OCR 預標報告")
    lines.append("產生時間：%s" % stats["generated_at"])
    lines.append("模型：%s" % model)
    lines.append("IoU 門檻：%.2f    目標綜合 F1：%.0f%%" % (iou_thr, target * 100))
    lines.append("")
    lines.append("===== 指標說明 =====")
    lines.append("IoU（框重疊度）：預標框與正解框『交集面積 ÷ 聯集面積』，"
                 "1.0=完全重合、0=完全沒重疊；>= 門檻(此處 %.2f) 才算『框對』。" % iou_thr)
    lines.append("1) 偵測率：只看框的位置、不管文字。"
                 "precision=框對÷你框的數；recall=框對÷正解數；F1=兩者調和平均。")
    lines.append("2) 辨識率：只在『已框對』的框裡，文字讀對的比例（=文字對÷框對），"
                 "不管漏框/多框。")
    lines.append("3) 綜合準確率：位置對(IoU>=門檻) 且 文字對，才算正確，再算 F1。"
                 "這是最終目標；綜合 ≈ 偵測 × 辨識。")
    lines.append("")
    lines.append("===== 計時 =====")
    lines.append("處理張數：%d" % timing["processed"])
    lines.append("每張平均：%.1f 秒（最快 %.1f，最慢 %.1f）" %
                 (timing["avg_seconds"], timing["min_seconds"], timing["max_seconds"]))
    lines.append("總花費：%.1f 秒（%.1f 分）" % (timing["total_seconds"], timing["total_seconds"] / 60.0))
    lines.append("Token：輸入 %d，輸出 %d，總計 %d（每張平均 入%d/出%d）" % (
        timing.get("tokens_in", 0), timing.get("tokens_out", 0), timing.get("tokens_total", 0),
        timing.get("avg_tokens_in", 0), timing.get("avg_tokens_out", 0)))
    lines.append("")
    lines.append("===== 每張 =====")
    lines.append("%-26s %7s %6s %6s %6s %6s %6s %7s %7s" %
                 ("檔名", "秒", "正解框", "框對", "多框", "漏框", "字對", "token入", "token出"))
    for pi in per_image:
        if pi.get("gt_available"):
            lines.append("%-26s %7.1f %6d %6d %6d %6d %6d %7d %7d" %
                         (pi["name"][:26], pi["seconds"], pi["gt_boxes"], pi["hit"],
                          pi["spurious"], pi["missed"], pi["text_ok"],
                          pi.get("tokens_in", 0), pi.get("tokens_out", 0)))
        else:
            lines.append("%-26s %7.1f %6s（無正解，未評分）" %
                         (pi["name"][:26], pi["seconds"], "-"))
    lines.append("")
    lines.append("===== 準確率 =====")
    lines.append("1) 偵測率  precision %.1f%% / recall %.1f%% / F1 %.1f%%（框對 %d，多框 %d，漏框 %d）" %
                 (dp * 100, dr * 100, df * 100, det_tp, det_fp, det_fn))
    lines.append("2) 辨識率  %.1f%%（框對的 %d 個裡，文字讀對 %d 個）" %
                 (reco_acc * 100, reco_total, reco_ok))
    lines.append("3) 綜合準確率（框對+字對）F1 %.1f%%（precision %.1f%% / recall %.1f%%）" %
                 (ef * 100, ep * 100, er * 100))
    lines.append(("✅ 已達標（>= %.0f%%）" % (target * 100)) if ef >= target
                 else ("⏳ 未達標，還差 %.1f 個百分點到 %.0f%%" % ((target - ef) * 100, target * 100)))
    lines.append("")
    if missed:
        lines.append("--- 漏框（該框卻沒框到）---")
        for fn, t in missed:
            lines.append("   [%s] 缺：%r" % (fn, t))
        lines.append("")
    if wrong_text:
        lines.append("--- 框到了但文字讀錯（pred → 應為）---")
        for fn, pt, gtt in wrong_text:
            lines.append("   [%s] %r → %r" % (fn, pt, gtt))
        lines.append("")
    if spurious:
        lines.append("--- 多框（框了正解沒有的）---")
        for fn, t in spurious:
            lines.append("   [%s] 多：%r" % (fn, t))
        lines.append("")
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # 終端機也印重點
    print("\n----- 準確率摘要 -----")
    print("偵測 F1 %.1f%% ／ 辨識 %.1f%% ／ 綜合 F1 %.1f%%%s" %
          (df * 100, reco_acc * 100, ef * 100,
           "  ✅達標" if ef >= target else "  ⏳未達標"))
    return stats


def append_run_log(log_dir, out_dir, model, stats):
    """把這一輪（含模型、token）追加到累積日誌，方便追蹤換模型的歷程。"""
    import csv
    path = os.path.join(log_dir, "gemini_runs_log.tsv")
    header = ["時間", "輸出資料夾", "模型", "張數", "每張秒", "總秒",
              "偵測F1%", "辨識%", "綜合F1%", "達標",
              "token入", "token出", "token總"]
    t = stats["timing"]
    m = stats["metrics"]
    row = [
        stats["generated_at"], os.path.basename(out_dir), model,
        t.get("processed"), t.get("avg_seconds"), t.get("total_seconds"),
        "%.1f" % (m["detection"]["f1"] * 100),
        "%.1f" % (m["recognition_accuracy"] * 100),
        "%.1f" % (m["end_to_end"]["f1"] * 100),
        "yes" if m["reached_target"] else "no",
        t.get("tokens_in", 0), t.get("tokens_out", 0), t.get("tokens_total", 0),
    ]
    # 若舊日誌沒有 token 欄，先升級：補上新表頭並把舊資料列補空白 token 欄
    old_rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            old_rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    needs_upgrade = bool(old_rows) and (len(old_rows[0]) < len(header))
    with open(path, "w" if (not old_rows or needs_upgrade) else "a",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if not old_rows:
            w.writerow(header)
        elif needs_upgrade:
            w.writerow(header)
            for r in old_rows[1:]:  # 跳過舊表頭
                w.writerow(r + [""] * (len(header) - len(r)))
        w.writerow(row)
    return path


def main():
    ap = argparse.ArgumentParser(description="用 Gemini 對發票做自動預標，輸出 roLabelImg XML")
    ap.add_argument("target", help="圖片檔或含圖片的資料夾")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemini 模型 (預設 %s)" % DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="只印出偵測結果，不寫檔")
    ap.add_argument("--outdir", metavar="DIR",
                    help="指定輸出資料夾；預設會自動在圖片資料夾裡新建一個帶時間戳的子資料夾")
    ap.add_argument("--workers", type=int, default=4,
                    help="同時併發處理的張數（預設 4；量大想快可調高，但太高會撞 API 速率限制 429）")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("錯誤：找不到金鑰。請先設定環境變數 GEMINI_API_KEY（或 GOOGLE_API_KEY）。")

    prompt = build_prompt()

    images = gather_images(args.target)
    if not images:
        sys.exit("在 %s 找不到圖片。" % args.target)

    # 決定輸出資料夾：
    # 為保護正解，預標『永遠』寫到獨立資料夾，絕不寫在圖片旁邊蓋到正解 XML。
    # 未指定 --outdir 時，自動在圖片資料夾裡新建帶時間戳的子資料夾。
    base_dir = args.target if os.path.isdir(args.target) else os.path.dirname(os.path.abspath(args.target))
    if args.outdir:
        out_dir = args.outdir
    else:
        out_dir = os.path.join(base_dir, "gemini_" + time.strftime("%Y%m%d_%H%M%S"))
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        print("輸出資料夾：%s" % out_dir)
        # 把這一輪實際使用的提示詞存進輸出資料夾，永久留存、可追溯
        with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
            pf.write("# 模型：%s\n# 時間：%s\n\n%s" %
                     (args.model, time.strftime("%Y-%m-%d %H:%M:%S"), prompt))

    # 詳細處理日誌：逐張寫入並即時 flush，跑到一半中斷也留得住
    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") if not args.dry_run else None

    def logline(msg=""):
        print(msg)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    from google import genai
    client = genai.Client(api_key=api_key)

    workers = max(1, args.workers)
    logline("處理日誌｜開始：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    logline("模型：%s （OCR 模式），共 %d 張圖，併發 %d\n" % (args.model, len(images), workers))
    per_image_secs = []
    results = []  # 每張：{name, seconds, boxes, pred_xml, tokens_in, tokens_out}
    tok_in = tok_out = tok_total = 0

    def process_one(img_path):
        """單張：讀尺寸 → 呼叫 Gemini（含重試）→ 回傳資料。可安全地被多執行緒呼叫。"""
        t0 = time.time()
        with Image.open(img_path) as im:
            width, height = im.size
            depth = len(im.getbands())
        detections, tok = detect(client, args.model, img_path, prompt)
        return {"img_path": img_path, "width": width, "height": height, "depth": depth,
                "detections": detections, "tok": tok, "elapsed": time.time() - t0}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    total = len(images)
    total_start = time.time()
    done = ok = fail = 0

    def prog():
        """進度前綴：完成數/總數、百分比、成功/失敗、已用與預估剩餘時間。"""
        el = time.time() - total_start
        eta = (el / done) * (total - done) if done else 0
        return "[%d/%d %.0f%%｜成功%d 失敗%d｜已用%.1f分｜預估剩%.1f分]" % (
            done, total, 100.0 * done / total, ok, fail, el / 60.0, eta / 60.0)

    logline("開始處理…（併發 %d）\n" % workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, p): p for p in images}
        for fut in as_completed(futures):
            p = futures[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                logline("%s ✗ 失敗：%s -> %s" % (prog(), os.path.basename(p), e))
                continue
            ok += 1
            tok = r["tok"]
            per_image_secs.append(r["elapsed"])
            tok_in += tok["in"]; tok_out += tok["out"]; tok_total += tok["total"]

            if args.dry_run:
                print("%s %s（%.1f 秒，token 入%d/出%d）" % (
                    prog(), os.path.basename(r["img_path"]), r["elapsed"],
                    tok["in"], tok["out"]))
                print(json.dumps(r["detections"], ensure_ascii=False, indent=2))
                continue

            out, kept = write_xml(r["img_path"], r["detections"], r["width"], r["height"],
                                  r["depth"], out_dir=out_dir)
            results.append({"name": os.path.basename(r["img_path"]),
                            "seconds": round(r["elapsed"], 2), "boxes": kept,
                            "pred_xml": out,
                            "tokens_in": tok["in"], "tokens_out": tok["out"]})
            logline("%s %s → %d框 %.1fs（token 入%d/出%d）" % (
                prog(), os.path.basename(r["img_path"]), kept, r["elapsed"],
                tok["in"], tok["out"]))

    processed = len(per_image_secs)
    total_elapsed = time.time() - total_start
    timing = {
        "processed": processed,
        "workers": workers,
        "total_seconds": round(total_elapsed, 2),
        "avg_seconds": round(sum(per_image_secs) / len(per_image_secs), 2) if per_image_secs else 0,
        "min_seconds": round(min(per_image_secs), 2) if per_image_secs else 0,
        "max_seconds": round(max(per_image_secs), 2) if per_image_secs else 0,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "tokens_total": tok_total,
        "avg_tokens_in": round(tok_in / processed) if processed else 0,
        "avg_tokens_out": round(tok_out / processed) if processed else 0,
    }
    logline("\n===== 計時統計 =====")
    logline("實際處理張數：%d（成功 %d，失敗 %d）" % (processed, ok, fail))
    if per_image_secs:
        logline("每張平均：%.1f 秒（最快 %.1f，最慢 %.1f）" % (
            timing["avg_seconds"], timing["min_seconds"], timing["max_seconds"]))
    logline("總花費時間：%.1f 秒（%.1f 分）＝實際牆鐘時間，併發 %d（每張延遲平均 %.1f 秒）" % (
        total_elapsed, total_elapsed / 60.0, workers, timing["avg_seconds"]))
    logline("Token：輸入 %d，輸出 %d，總計 %d（每張平均 入%d/出%d）" % (
        tok_in, tok_out, tok_total, timing["avg_tokens_in"], timing["avg_tokens_out"]))

    if not args.dry_run:
        logline("預標 XML 已寫到：%s" % out_dir)
        # 檢查圖片資料夾最上層是否有同名正解 XML
        has_gt = any(os.path.exists(os.path.join(base_dir,
                     os.path.splitext(r["name"])[0] + ".xml")) for r in results)
        if has_gt:
            stats = write_reports(out_dir, base_dir, results, timing, args.model)
            logline("統計檔已寫到：%s/report.txt 與 stats.json" % out_dir)
            log_path = append_run_log(base_dir, out_dir, args.model, stats)
            logline("累積日誌已更新：%s" % log_path)
        else:
            write_timing_only_report(out_dir, results, timing, args.model)
            logline("統計檔已寫到：%s/report.txt 與 stats.json（未評分）" % out_dir)
        logline("處理日誌已寫到：%s/run.log" % out_dir)
    logline("\n完成。（正解 XML 未被更動）")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
