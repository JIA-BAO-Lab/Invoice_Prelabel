# invoice-prelabel

用 Google Gemini 對台灣統一發票做「全文 OCR 自動預標」，輸出 **PascalVOC XML**（roLabelImg 等標注工具可直接開），並自動計時、統計 token、比對正解算準確率。

本工具**獨立運作，不依賴 roLabelImg 應用程式**；僅內含一份 `pascal_voc_io.py`（複製自 labelImg/roLabelImg，MIT 授權）負責讀寫 XML，因此輸出格式與 roLabelImg 完全相容。

## 檔案

| 檔案 | 用途 |
|------|------|
| `gemini_prelabel.py` | 預標主程式：呼叫 Gemini → 產生 XML（robndbox）→ 自動比對正解 → 產 report/stats/log |
| `eval_prelabel.py` | 獨立評分工具（偵測率／辨識率／綜合準確率）|
| `preview_boxes.py` | 把框疊到圖上輸出預覽 PNG（無正解時目視檢查用）|
| `invoice_prompt.txt` | 目前使用的提示詞（實際提示詞寫在 `gemini_prelabel.py` 的 `build_prompt()`）|
| `pascal_voc_io.py` | PascalVOC XML 讀寫（內含，MIT）|
| `requirements.txt` | 相依套件：google-genai / Pillow / lxml |

## 安裝

```bash
cd /Users/kevinchan/git/invoice-prelabel
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## 設定金鑰

金鑰只放在本機檔案，不進程式碼、不進 git：

```bash
echo 'export GEMINI_API_KEY=你的金鑰' > ~/.gemini_env && chmod 600 ~/.gemini_env
```

執行前先載入：`source ~/.gemini_env`

## 使用

### 1) 預標一個資料夾的發票圖

```bash
source ~/.gemini_env
./.venv/bin/python gemini_prelabel.py <圖片資料夾> --model gemini-3.8-flash
```

- 會在圖片資料夾裡自動新建 `gemini_<時間戳>/` 子夾，放每張的 `.xml`、`run.log`（逐張處理日誌，即時寫入）、`report.txt`、`stats.json`、`prompt.txt`。
- **絕不覆蓋圖片資料夾最上層的正解 XML。**
- 若最上層有同名正解 XML → 自動比對算準確率，並把該輪追加到 `gemini_runs_log.tsv`（含模型、準確率、耗時、token）。
- 若沒有正解 → 自動跳過評分，只出計時 + token。
- 選項：`--dry-run`（只印不寫檔）、`--outdir <資料夾>`（自訂輸出位置）、`--workers N`（同時併發 N 張，預設 4）。

**大量處理要快**：加 `--workers` 併發送多張，速度接近線性提升。例如 10 張序列 ~72 秒，`--workers 5` 約 18 秒（~4 倍）。量大時可調到 8~10，但太高會撞 API 速率限制（429）；遇到 429 就把 workers 調低。品質不受併發影響。

```bash
./.venv/bin/python gemini_prelabel.py <圖片資料夾> --model gemini-3.8-flash --workers 8
```

### 1b) 批次模式（非同步、半價，適合大量過夜）— `gemini_batch.py`

走 Gemini Batch API：價格約即時模式的**一半**，但**非即時**（送出後排隊，數分鐘到 24 小時完成）。**兩段式**，送出後可關終端機、隔天再取回。與即時模式並存、輸出格式完全一致。

```bash
# 1) 送出（馬上結束，會給你一個「批次資料夾」）
./.venv/bin/python gemini_batch.py submit "<圖片資料夾>" --model gemini-3.8-flash

# 2) 稍後取回（可關終端機、隔天再來；沒好會叫你等，跑同一行即可）
./.venv/bin/python gemini_batch.py fetch "<批次資料夾>"
```

- 送出時在圖片資料夾建 `gemini_batch_<時間戳>/`，內含 `batch_job.json`（工作代號、圖片順序與尺寸）與 `prompt.txt`。
- 取回完成後，同資料夾產生 XML、`run.log`、`report.txt`、`stats.json`（有正解一樣自動比對準確率並記入 `gemini_runs_log.tsv`）。
- `--chunk-size N`：每個批次工作最多幾張（預設 200），大量時自動分成多個工作。
- **自動取回（可選）**：`fetch --wait` 會開著每隔一段時間自動查、完成才取回；`--interval 30` 設幾分鐘查一次（預設 30）。此模式**需保持終端開啟**；關掉或關機也沒關係——之後再手動 `fetch` 一次即可（批次在 Google 端照跑，不受影響）。預設是**手動**（不加 `--wait`）。
  ```bash
  ./.venv/bin/python gemini_batch.py fetch "<批次資料夾>" --wait --interval 30
  ```
- **何時用**：趕時間/邊看邊修 → 即時 `gemini_prelabel.py`；量大、不趕、想省一半 → 批次。

### 2) 單獨評分（比對預標 vs 正解）

```bash
./.venv/bin/python eval_prelabel.py --pred <預標XML資料夾> --gt <正解XML資料夾> [--iou 0.5]
```

### 3) 疊圖預覽（無正解時目視檢查）

```bash
./.venv/bin/python preview_boxes.py <圖片資料夾> <預標XML資料夾> [輸出資料夾]
```

## 三個準確率指標

- **偵測率**：只看框的位置，IoU ≥ 門檻(0.5) 即算框對。precision/recall/F1。
- **辨識率**：在「已框對」的框裡，文字讀對的比例。
- **綜合準確率**：位置對且文字對，才算正確（最終目標）。綜合 ≈ 偵測 × 辨識。

## 模型建議

| 模型 | 綜合 F1(本專案) | 速度 | 備註 |
|------|----------------|------|------|
| gemini-2.5-flash | ~7% | 慢 | 不建議 |
| gemini-3.1-flash-lite | ~40% | 快 | 小框定位不足 |
| gemini-3.1-pro-preview | ~88% | 中 | 可 |
| **gemini-3.8-flash** | **~88–90%** | **最快** | **推薦（CP 值最高）** |

## 標注慣例

label 填「實際讀到的文字」（非欄位類型）。詳見 `invoice_prompt.txt`：框發票號碼(字軌與數字分開)、日期、統編(表格內逐字/印章內整串)、表頭字、金額、大寫金額、✓ 等；不框公司名稱/地址/品項/印章說明字。

## 費用（gemini-3.8-flash，每張約 2214 入 / 1179 出 token）

以官方優惠價（~2026/12/31）估：每張約 $0.006、每千張約 $6、每四千張約 $24 美元。2027 起標準價翻倍；Batch 批次模式可再半價。
