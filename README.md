# อิทธิพลของหุ้น Top 10 (market cap) ต่อ S&P 500

ศึกษาว่าหุ้น 10 อันดับแรกตาม market cap ของ S&P 500 (ซึ่งเปลี่ยนตัวได้ตลอดเวลา) อธิบายการเคลื่อนไหวรายวันของดัชนีได้มากแค่ไหน
และอันดับไหนสำคัญที่สุด — ด้วย Linear/Ridge, XGBoost, SVR, LSTM และ AutoGluon พร้อม feature importance (permutation, SHAP)

ข้อมูล: ม.ค. 2016 – ธ.ค. 2025 (2,514 วันซื้อขาย) · train 2,011 วัน / test 503 วัน (ธ.ค. 2023 – ธ.ค. 2025) · expanding-window CV 4 fold × 365 วัน

## ผลหลัก (test 2024–2025)

| Model | R² | MAE (จุด) | MSE (จุด²) | RMSE (จุด) | MAPE (%) |
|---|---|---|---|---|---|
| AutoGluon | 0.825 | 18.4 | 582 | 24.1 | 0.317 |
| XGBoost | 0.813 | 19.3 | 638 | 25.3 | 0.332 |
| SVR | 0.780 | 20.0 | 711 | 26.7 | 0.346 |
| Ridge | 0.748 | 22.1 | 847 | 29.1 | 0.383 |
| LSTM | 0.712 | 21.0 | 896 | 29.9 | 0.364 |
| Linear | 0.707 | 23.7 | 979 | 31.3 | 0.409 |

R² คำนวณบน return รายวัน; MAE/MSE/RMSE/MAPE คำนวณบนระดับดัชนี (ระดับที่ทำนาย = ระดับเมื่อวาน × (1 + return ที่ทำนาย))

- Top 10 อธิบาย return รายวันของ S&P 500 ได้ ~80% (CV 0.78–0.80 ทุกโมเดล tabular; สูงสุดในปีวิกฤต 2018/2020/2022 = 0.89–0.93)
- Sensitivity ของดัชนีต่อ top 10 ขึ้นกับ regime: ~0.7–0.9% ต่อ 1% ในปีผันผวน แต่ ~0.5% ในปี 2023–24
- Importance กระจุกที่อันดับ 7–10 ไม่ใช่ 1–3 — เป็นผลของ collinearity ในกลุ่มเทค (อันดับ 1–5 แชร์เครดิต) และอันดับ 7–10 เป็น proxy ของหุ้นอีก 490 ตัว ไม่ใช่น้ำหนัก market cap
- โมเดล non-linear (AutoGluon/XGBoost) ดีกว่า Linear บน test อย่างมีนัยสำคัญ (block bootstrap + Diebold–Mariano) แต่ใน fold COVID Linear ดีที่สุดเพราะ extrapolate ได้

รายละเอียด การตรวจสอบ (ADF, next-day placebo, sensitivity ของ un-swap, VIF) และคำถาม-คำตอบสำหรับการนำเสนออยู่ใน notebook section 14–15

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `sp500_top10_influence.ipynb` | notebook หลัก (รันแล้ว มี output ครบ) — data prep, un-swap, โมเดล 6 ตัว, tuning, importance, สรุปผล, Q&A |
| `dashboard.py` | Streamlit dashboard 7 หน้า อ่านผลจาก `dashboard_data.pkl` |
| `dashboard_data.pkl` | ผลลัพธ์ทั้งหมดที่ notebook export (metrics, predictions, importance, diagnostics ฯลฯ) |
| `data_prep.py` | โมดูลเวอร์ชันของ section 3–4 (un-swap + split) สำหรับใช้ในสคริปต์ |
| `modelling_table(new).csv` | ข้อมูลต้นทาง: `close_1..close_10` = adj close ของหุ้นอันดับ 1–10 *ในวันนั้น* (ช่องอันดับ ไม่ใช่หุ้นตัวเดิม), `target` = S&P 500 |
| `results_summary.csv` | ตาราง metric ทุกโมเดล ทุก split (export จาก `dashboard_data.pkl`) |

## วิธีรัน

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Dashboard (ใช้ผลที่ export ไว้แล้ว ไม่ต้องรัน notebook):

```bash
.\.venv\Scripts\python -m streamlit run dashboard.py
```

รัน notebook ใหม่ (~3 นาทีบน CPU; AutoGluon เพิ่ม ~10 นาที — ตั้ง `RUN_AUTOGLUON = False` ที่ cell แรกถ้าไม่ได้ติดตั้ง):

```bash
.\.venv\Scripts\jupyter lab sp500_top10_influence.ipynb
```

## จุดสำคัญของวิธีการ

1. **ใช้ return ไม่ใช่ level** — level ไม่ stationary (ADF p ≈ 0.98) ทำให้ R² 0.99 แบบ spurious และ tree model extrapolate ไม่ได้ (test R² = −4.1)
2. **Un-swap** — คอลัมน์เป็นช่องอันดับ ต้องจับคู่ราคาวัน t กับ t−1 ให้เป็นบริษัทเดิมก่อนคำนวณ return (Hungarian assignment + rank penalty + market proxy + กฎหุ้นเข้า/ออกที่อันดับ 9–10); ตรวจสอบด้วยคู่ราคาใกล้กัน (corr 0.976), R² ก่อน/หลัง (−1.9 → 0.71) และ sensitivity ของพารามิเตอร์
3. **Same-day** — วัดอิทธิพล ไม่ใช่การพยากรณ์ (ใช้ return เมื่อวานทำนายวันนี้ได้ R² ≈ 0)
4. **ไม่มี leakage** — scaler fit ต่อ fold, hyperparameter เลือกจาก CV เท่านั้น, LSTM/AutoGluon ใช้ส่วนท้ายของ train สำหรับ early stopping/tuning, test วัดครั้งเดียว

ข้อจำกัด: ไม่มี ticker และ market cap → ตีความได้ระดับอันดับ และ coefficient/importance เป็น sensitivity เชิงสถิติ (รวม co-movement) ไม่ใช่น้ำหนักในดัชนี
