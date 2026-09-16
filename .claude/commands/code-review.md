---
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(git status:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh pr list:*), Bash(gh pr comment:*)
description: 對這個發票預標專案的變更做高訊號 code review
---

對指定的變更做 code review。目標可為：PR 編號、分支名、或不指定（預設審目前工作區相對 `main` 的差異）。

**核心原則：只回報「高把握、高價值」的問題。** 寧可漏報，不要誤報——誤報會浪費 reviewer 時間、侵蝕信任。

## 步驟

1. **確定審查目標並取得 diff**
   - 有 PR 編號 → `gh pr diff <PR>`；有分支 → `git diff main...<branch>`；都沒有 → `git diff main...HEAD`（或 `git diff` 工作區）。
   - 先讀 PR/commit 標題與描述，理解作者意圖。

2. **只看改動本身**：聚焦 diff，不要把整個 codebase 的既有問題（pre-existing）算進來。

3. **逐面向審查**（見下方檢查清單），彙整成一份問題列表，每則含：檔案:行、問題描述、為什麼是問題（會導致什麼後果）、建議修法。

4. **驗證每一則**：能不能明確指出「在什麼輸入/情況下會出錯」？不能就刪掉。無法在 diff 範圍內確認的，不要報。

5. **輸出**（見下方格式）。若帶 `--comment` 且審 PR：用 `gh pr comment` 貼一則彙整留言（沒問題也貼「無問題」）。

## 只回報這些（高訊號）

- **會壞的錯**：語法/型別錯、未定義變數、缺 import、必定跑錯的邏輯（與輸入無關）。
- **本專案關鍵不變量被破壞**（見下）。
- **安全問題**：金鑰/機敏外洩、把使用者資料送到非預期位置。

## 不要報這些（誤報來源）

- 純風格、命名、格式（linter 的事）。
- 「某些輸入下也許會…」這種要靠 diff 外脈絡才能判斷的臆測。
- 主觀的「可以更好」建議、測試覆蓋率不足（除非明確要求）。
- 既有的、非本次改動造成的問題。

## 本專案關鍵不變量（務必檢查改動有沒有破壞）

1. **金鑰安全**：金鑰只能從環境變數（`GEMINI_API_KEY` / `GOOGLE_API_KEY`）讀；**絕不可**寫進檔案、log、commit、或印到終端機（連片段都不行）。`prompt.txt` / `run.log` / `report` 不得含金鑰。
2. **正解不可覆蓋**：預標輸出一律寫到圖片資料夾裡的 `gemini_*/` 子夾；**永遠不得覆蓋圖片資料夾最上層的正解 XML**。任何改到輸出路徑的地方都要確認這點。
3. **座標換算**：Gemini `box_2d` 是 `[ymin, xmin, ymax, xmax]` 正規化 0–1000；轉像素要對應正確的 width/height，並夾在圖片範圍內。搞混 x/y 或 width/height 是嚴重 bug。
4. **輸出格式**：XML 由內含的 `pascal_voc_io.py` 產生（robndbox），需與 roLabelImg 相容；不要自行手拼 XML。
5. **穩健性**：API 呼叫要保留重試；模型回傳可能是壞掉/截斷的 JSON，解析要保留 `extract_json_array` 的搶救邏輯，單張失敗不可讓整批中斷。
6. **並行安全**（`gemini_prelabel.py`）：多執行緒下，共享狀態的累加要正確；每張寫到各自的檔案路徑。
7. **批次對應**（`gemini_batch.py`）：`fetch` 靠**送出順序**把回應對回每張圖（`InlinedResponse` 無 metadata）；改動 submit 的順序/分塊或 fetch 的索引對應時要特別小心。單一回應的 `error` 要容錯、不可中斷整批。
8. **評分定義**：偵測率（IoU≥門檻）、辨識率（框對之中文字對的比例）、綜合（框對且字對）——動到 `eval_prelabel.py` 的比對/正規化邏輯時，確認三指標定義不被破壞。

## 輸出格式

```
## Code review

發現 N 個問題（或：No issues found.）

1. [檔案:行] 問題描述
   - 為什麼：在 <情況> 會 <後果>
   - 建議：<修法>
...
```

若無問題：`No issues found. 已檢查：bug、專案不變量、安全。`
