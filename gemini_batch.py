#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_batch.py — 批次(非同步)發票 OCR 預標，兩段式：submit / fetch。

批次模式走 Gemini Batch API：價格約即時模式的一半，但非即時（送出後排隊，
可能數分鐘到 24 小時才完成）。適合大量、過夜處理。

與即時模式 (gemini_prelabel.py) 並存、互不影響；輸出格式完全一致
（robndbox XML + report.txt + stats.json + run.log，有正解則自動比對準確率）。

用法：
    # 1) 送出（馬上結束，給你一個批次資料夾）
    python gemini_batch.py submit <圖片資料夾> --model gemini-3.8-flash

    # 2) 稍後取回（可關終端機、隔天再來）
    python gemini_batch.py fetch <批次資料夾>

送出時會在圖片資料夾裡新建 gemini_batch_<時間戳>/，內含 batch_job.json（工作代號、
圖片順序與尺寸）與 prompt.txt。fetch 就是對這個資料夾操作。
"""
import argparse
import datetime
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemini_prelabel import (  # noqa: E402
    DEFAULT_MODEL, build_prompt, gather_images, extract_json_array,
    write_xml, write_reports, write_timing_only_report, append_run_log,
)

HANDLE = "batch_job.json"


def _mime(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext in ("tif", "tiff"):
        return "image/tiff"
    return "image/" + ext


def _api_key():
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        sys.exit("錯誤：找不到金鑰。請先設定環境變數 GEMINI_API_KEY（或 GOOGLE_API_KEY）。")
    return k


def cmd_submit(args):
    from google import genai
    from google.genai import types

    prompt = build_prompt()
    images = gather_images(args.target)
    if not images:
        sys.exit("在 %s 找不到圖片。" % args.target)

    base_dir = args.target if os.path.isdir(args.target) \
        else os.path.dirname(os.path.abspath(args.target))
    out_dir = args.outdir or os.path.join(base_dir, "gemini_batch_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
        pf.write("# 模型：%s（批次）\n# 時間：%s\n\n%s" %
                 (args.model, time.strftime("%Y-%m-%d %H:%M:%S"), prompt))

    client = genai.Client(api_key=_api_key())
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                      max_output_tokens=8192)

    # 讀尺寸 + 準備 inlined 請求（順序＝之後回應對應的順序）
    meta_images = []
    reqs = []
    for p in images:
        with Image.open(p) as im:
            w, h = im.size
            depth = len(im.getbands())
        with open(p, "rb") as f:
            data = f.read()
        meta_images.append({"name": os.path.basename(p), "width": w, "height": h, "depth": depth})
        reqs.append(types.InlinedRequest(
            model=args.model,
            contents=[types.Part.from_bytes(data=data, mime_type=_mime(p)), prompt],
            config=cfg,
        ))

    # 分塊送出（每塊一個批次工作），避免單一請求過大
    chunk = max(1, args.chunk_size)
    jobs = []
    print("送出 %d 張，分成 %d 塊（每塊最多 %d 張）…" %
          (len(reqs), (len(reqs) + chunk - 1) // chunk, chunk))
    for start in range(0, len(reqs), chunk):
        part = reqs[start:start + chunk]
        job = client.batches.create(
            model=args.model, src=part,
            config=types.CreateBatchJobConfig(
                display_name="invoice-prelabel-%s-%d" % (time.strftime("%Y%m%d_%H%M%S"), start)),
        )
        jobs.append({"job_name": job.name, "start": start, "count": len(part),
                     "state": str(job.state)})
        print("  ✓ 第 %d-%d 張 → %s（%s）" %
              (start, start + len(part) - 1, job.name, job.state))

    handle = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "base_dir": base_dir,
        "images": meta_images,
        "jobs": jobs,
    }
    with open(os.path.join(out_dir, HANDLE), "w", encoding="utf-8") as f:
        json.dump(handle, f, ensure_ascii=False, indent=2)

    print("\n批次已送出。批次資料夾：%s" % out_dir)
    print("稍後（可關終端機/隔天）用這行取回結果：")
    print('  python gemini_batch.py fetch "%s"' % out_dir)


def _state_kind(state):
    s = str(state).upper()
    if "SUCCEEDED" in s:
        return "ok"
    if "FAILED" in s or "CANCELLED" in s or "EXPIRED" in s:
        return "bad"
    return "pending"


def cmd_fetch(args):
    from google import genai

    out_dir = args.batch_dir
    hp = os.path.join(out_dir, HANDLE)
    if not os.path.exists(hp):
        sys.exit("在 %s 找不到 %s（請指向 submit 產生的批次資料夾）。" % (out_dir, HANDLE))
    handle = json.load(open(hp, encoding="utf-8"))
    base_dir = handle["base_dir"]
    model = handle["model"]
    images = handle["images"]

    client = genai.Client(api_key=_api_key())
    jobs = handle["jobs"]

    # 查所有子工作狀態；--wait 開啟時，未完成就每隔 interval 分鐘自動再查
    while True:
        states = [(j, client.batches.get(name=j["job_name"])) for j in jobs]
        kinds = [_state_kind(job.state) for _, job in states]
        print("批次工作：共 %d 個，完成 %d，處理中 %d，失敗 %d" %
              (len(jobs), kinds.count("ok"), kinds.count("pending"), kinds.count("bad")))
        for (j, job) in states:
            print("  - %s：%s" % (j["job_name"], job.state))
        if "pending" not in kinds:
            break
        if not args.wait:
            print("\n⏳ 還有工作在處理中，尚未全部完成。稍後再執行同一行 fetch 即可。")
            return
        print("\n⏳ 尚未完成，%d 分鐘後自動再查…（Ctrl+C 可中止；直接關掉也沒關係，"
              "之後再手動 fetch 即可）\n" % args.interval)
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n已中止自動輪詢。之後再手動執行 fetch 取回即可。")
            return

    # 全部結束（可能含失敗工作）→ 收集回應，依 handle 的順序對回每張圖
    responses = [None] * len(images)
    for (j, job) in states:
        dest = getattr(job, "dest", None)
        inl = getattr(dest, "inlined_responses", None) if dest else None
        if not inl:
            print("  ⚠ 工作 %s 沒有回應（狀態 %s）" % (j["job_name"], job.state))
            continue
        if len(inl) != j["count"]:
            print("  ⚠ 工作 %s 回應數(%d) 與送出張數(%d) 不符，該塊可能需重送（依序對應可能錯位）"
                  % (j["job_name"], len(inl), j["count"]))
        for k, ir in enumerate(inl):
            idx = j["start"] + k
            if 0 <= idx < len(images):
                responses[idx] = ir

    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8")

    def logline(msg=""):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    logline("批次取回｜時間：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logline("模型：%s（批次），送出於 %s，共 %d 張\n" % (model, handle["created_at"], len(images)))

    results = []
    tok_in = tok_out = tok_total = 0
    ok = fail = 0
    for i, meta in enumerate(images):
        name = meta["name"]
        ir = responses[i]
        if ir is None or getattr(ir, "error", None):
            fail += 1
            err = getattr(ir, "error", "無回應") if ir else "無回應"
            logline("[%d/%d] ✗ 失敗：%s -> %s" % (i + 1, len(images), name, err))
            continue
        resp = ir.response
        try:
            detections = extract_json_array(resp.text)
        except Exception as e:  # noqa: BLE001
            fail += 1
            logline("[%d/%d] ✗ 解析失敗：%s -> %s" % (i + 1, len(images), name, e))
            continue
        u = getattr(resp, "usage_metadata", None)
        ti = getattr(u, "prompt_token_count", 0) or 0
        to = getattr(u, "candidates_token_count", 0) or 0
        tt = getattr(u, "total_token_count", 0) or 0
        tok_in += ti; tok_out += to; tok_total += tt
        img_path = os.path.join(base_dir, name)
        out, kept = write_xml(img_path, detections, meta["width"], meta["height"],
                              meta["depth"], out_dir=out_dir)
        results.append({"name": name, "seconds": 0, "boxes": kept, "pred_xml": out,
                        "tokens_in": ti, "tokens_out": to})
        ok += 1
        logline("[%d/%d] %s → %d框（token 入%d/出%d）" %
                (i + 1, len(images), name, kept, ti, to))

    # 批次歷時（送出→取回）
    try:
        created = datetime.datetime.strptime(handle["created_at"], "%Y-%m-%d %H:%M:%S")
        turnaround = (datetime.datetime.now() - created).total_seconds()
    except Exception:  # noqa: BLE001
        turnaround = 0
    timing = {
        "processed": ok, "workers": 0, "mode": "batch",
        "total_seconds": round(turnaround, 2),
        "avg_seconds": 0, "min_seconds": 0, "max_seconds": 0,
        "tokens_in": tok_in, "tokens_out": tok_out, "tokens_total": tok_total,
        "avg_tokens_in": round(tok_in / ok) if ok else 0,
        "avg_tokens_out": round(tok_out / ok) if ok else 0,
    }
    logline("\n===== 批次結果 =====")
    logline("成功 %d，失敗 %d（共 %d 張）" % (ok, fail, len(images)))
    logline("批次歷時（送出→取回）：%.1f 分" % (turnaround / 60.0))
    logline("Token：輸入 %d，輸出 %d，總計 %d（每張平均 入%d/出%d）" %
            (tok_in, tok_out, tok_total, timing["avg_tokens_in"], timing["avg_tokens_out"]))
    logf.close()

    # 報告 + 準確率（有正解才比對）
    has_gt = any(os.path.exists(os.path.join(base_dir, os.path.splitext(r["name"])[0] + ".xml"))
                 for r in results)
    if has_gt:
        stats = write_reports(out_dir, base_dir, results, timing, model + "（批次）")
        append_run_log(base_dir, out_dir, model + "（批次）", stats)
        print("統計檔已寫到：%s/report.txt 與 stats.json；準確率已比對並記入 gemini_runs_log.tsv" % out_dir)
    else:
        write_timing_only_report(out_dir, results, timing, model + "（批次）")
        print("統計檔已寫到：%s/report.txt 與 stats.json（無正解，未評分）" % out_dir)
    print("預標 XML + run.log 已寫到：%s" % out_dir)


def main():
    ap = argparse.ArgumentParser(description="批次(非同步)發票 OCR 預標：submit / fetch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="送出批次（馬上結束）")
    s.add_argument("target", help="圖片檔或含圖片的資料夾")
    s.add_argument("--model", default=DEFAULT_MODEL, help="Gemini 模型 (預設 %s)" % DEFAULT_MODEL)
    s.add_argument("--outdir", metavar="DIR", help="自訂批次資料夾（預設自動建時間戳子夾）")
    s.add_argument("--chunk-size", type=int, default=30, help="每個批次工作最多幾張（預設 30；避免 inlined 請求過大）")
    s.set_defaults(func=cmd_submit)

    f = sub.add_parser("fetch", help="取回批次結果")
    f.add_argument("batch_dir", help="submit 產生的批次資料夾（含 batch_job.json）")
    f.add_argument("--wait", action="store_true",
                   help="開著自動輪詢，每隔一段時間查一次、完成才取回（需保持終端開啟）")
    f.add_argument("--interval", type=int, default=30,
                   help="--wait 時每隔幾分鐘查一次（預設 30）")
    f.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
