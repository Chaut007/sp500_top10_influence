"""
data_prep.py — เตรียมข้อมูลกลางสำหรับสคริปต์ (เวอร์ชันโมดูลของ section 3-4 ใน sp500_top10_influence.ipynb)

ทำ 3 อย่าง
  1. โหลด modelling_table(new).csv  (ราคา adj close ของหุ้นอันดับ 1-10 ตาม market cap + ระดับ S&P 500)
  2. แปลงราคาเป็น return รายวัน แบบ "un-swap":
        r_k(t) = return วันนี้ของบริษัทที่อยู่อันดับ k ในวัน t
        - จับคู่ค่าวัน t กับวัน t-1 แบบ optimal assignment (Hungarian) ด้วย cost = |log return - market proxy| (capped 35%)
          + penalty 1% ต่อการเลื่อน 1 อันดับ  -> แก้ปัญหาคู่ราคาใกล้กัน (GOOGL/GOOG) และวันตลาดผันผวน
        - market proxy = median ของ return 10 ช่อง (ไม่ใช้ target -> ไม่ leak)
        - หุ้นเข้า/ออก top 10: |log return| > 35%; หรือที่อันดับ 9-10 idiosyncratic move > 20%; หรือ 8-20% ที่มีหลักฐาน
          (ตัวที่เข้ามาตรงกับหุ้นที่เพิ่งออกไป / หุ้นที่ออกไปกลับมาเร็วๆ นี้)  -> NaN -> เติมด้วยค่าเฉลี่ยของช่องอื่นวันนั้น
        - แถวแรก (4/1/2016) มี return ด้วย โดยใช้แถวอ้างอิง 31/12/2015 จาก Yahoo Finance (REF_ROW)
  3. สร้าง split ตามที่ตกลง:
        train 2,011 แถว (แถว 1-2,011) / test 503 แถว (แถว 2,012-2,514)
        expanding window 4 fold ใน train:
            fold1  train 1-551    val 552-916
            fold2  train 1-916    val 917-1,281
            fold3  train 1-1,281  val 1,282-1,646
            fold4  train 1-1,646  val 1,647-2,011
        (เลขแถวในโน้ตเป็น 1-based, ในโค้ดใช้ 0-based index ของ DataFrame)

วิธีใช้ในสคริปต์โมเดล
    from data_prep import build_dataset, get_folds, get_test, get_xy, RET_COLS, TARGET_RET
    df, diag = build_dataset()
    for fold in get_folds():
        X_tr, y_tr = get_xy(df, fold["train"])
        X_va, y_va = get_xy(df, fold["val"])
        ...
    test = get_test()

รันไฟล์นี้ตรงๆ (python data_prep.py) จะพิมพ์สรุป (ไม่เขียนไฟล์ — ผลลัพธ์ของโปรเจกต์เก็บใน dashboard_data.pkl จาก notebook)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CSV_PATH = Path(__file__).with_name("modelling_table(new).csv")

# แถวอ้างอิง 31/12/2015 (วันซื้อขายก่อนแถวแรกของไฟล์) จาก Yahoo Finance (yfinance auto_adjust=True, ดึง 14/9/2026)
# = adj close ของ GOOGL, GOOG, MSFT, AAPL, AMZN, XOM, BRK-B, META, JNJ, VZ ตามลำดับช่อง 1-10 ของวันที่ 4/1/2016 และ ^GSPC
# ใช้เพื่อคำนวณ return ของแถวแรกเท่านั้น (ไม่นับเป็นข้อมูล) — ตรวจแล้วว่าฐานการปรับราคาตรงกับไฟล์ (adj close 4/1/2016 ตรงกันที่ 1e-7)
REF_ROW = {"date": "2015-12-31", "close_1": 38.53383255, "close_2": 37.58797455, "close_3": 48.27262497, "close_4": 23.66843796,
           "close_5": 33.79449844, "close_6": 49.31230927, "close_7": 132.03999329, "close_8": 103.74891663, "close_9": 76.63905334,
           "close_10": 26.11704063, "target": 2043.939941}

SLOT_COLS = [f"close_{k}" for k in range(1, 11)]
RET_COLS = [f"r_{k}" for k in range(1, 11)]
TARGET_RET = "r_target"

N_ROWS = 2514
# (train_end, val_end) แบบ 0-based half-open: train = [0, train_end), val = [train_end, val_end)
FOLD_BOUNDS = [(551, 916), (916, 1281), (1281, 1646), (1646, 2011)]
TEST_START = 2011  # แถว 2,012 ในโน้ต


# ----------------------------------------------------------------------------- โหลด
def load_prices(path: Path | str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.set_index("date").sort_index()
    if len(df) != N_ROWS:
        raise ValueError(f"คาดว่ามี {N_ROWS} แถว แต่ได้ {len(df)} — ถ้าเปลี่ยนไฟล์ต้องปรับ FOLD_BOUNDS/TEST_START ด้วย")
    return df[SLOT_COLS + ["target"]]


# ----------------------------------------------------------------------------- un-swap
def unswap_returns(prices: pd.DataFrame, lam: float = 0.01, cap: float = 0.35, th_idio: float = 0.08,
                   th_big: float = 0.20, boundary: int = 8, dormant_days: int = 30, look_days: int = 20):
    '''แปลงตารางราคาแบบ "ช่องอันดับ" ให้เป็น return รายวันของบริษัทเดิม

    ทำทีละวัน t เทียบกับ t-1
      ขั้น 1  จับคู่ช่องวันนี้กับช่องเมื่อวานด้วย Hungarian assignment
              cost = |log return - market shift| (capped ที่ `cap`) + `lam` x จำนวนอันดับที่เลื่อน
              ทำ 2 รอบ: รอบแรกประมาณ market shift (median ของ log return) แล้วจับคู่ใหม่โดยหัก shift นั้น
      ขั้น 2  market proxy ของวัน = median ของ log return หลังจับคู่ (ไม่ใช้ target -> ไม่ leak)
      ขั้น 3  ตัดสินทุกคู่ว่าเป็นการขยับจริง ('real') หรือหุ้นเข้า/ออก ('exit' -> return เป็น NaN)
                - |log return| > cap                                          -> exit
                - ช่อง >= boundary (อันดับ 9-10) และ idiosyncratic move > th_big -> exit
                - ช่อง >= boundary และ idiosyncratic move > th_idio           -> exit เมื่อมีหลักฐานอย่างใดอย่างหนึ่ง
                     (ก) ราคาตัวที่เข้ามาตรงกับหุ้นที่ออกไปภายใน `dormant_days` (ปรับด้วย market proxy)
                     (ข) หุ้นที่ออกไปกลับมาภายใน `look_days` วัน และตัวที่เข้ามาหายไป
                - นอกนั้น                                                     -> real
    คืน (returns, diag, market_proxy)
      returns      : DataFrame r_1..r_10 (simple return; NaN = หุ้นเข้า/ออก หรือแถวแรก) + swapped + n_unmatched
      diag         : ทุก event ที่ idiosyncratic move > th_idio หรือ |log return| > cap พร้อม status และเหตุผล
      market_proxy : Series market proxy รายวัน (หน่วย %)
    '''
    log_p = np.log(prices[SLOT_COLS].to_numpy(dtype=float))              # (T วัน, K ช่อง)
    T, K = log_p.shape
    rank_penalty = lam * np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])

    def match_slots(t, market_shift):
        '''prev_slot[i] = ช่องของเมื่อวานที่จับคู่กับช่อง i ของวันนี้ (cost รวมต่ำสุด)'''
        cost = np.minimum(np.abs(log_p[t][:, None] - log_p[t - 1][None, :] - market_shift), cap) + rank_penalty
        row, col = linear_sum_assignment(cost)
        prev_slot = np.empty(K, int)
        prev_slot[row] = col
        return prev_slot

    def tol(days_gap):
        '''ความคลาดเคลื่อนของ log price ที่ยอมรับเมื่อเทียบราคาข้ามวัน — กว้างขึ้นตาม sqrt(จำนวนวัน)'''
        return 0.02 + 0.02 * np.sqrt(max(days_gap, 1))

    # ---- ขั้น 1-2: จับคู่ทุกวัน + market proxy ----
    prev_slot = np.zeros((T, K), int)         # ช่องเมื่อวานของแต่ละช่องวันนี้
    log_ret = np.full((T, K), np.nan)         # log return หลังจับคู่
    mkt = np.zeros(T)                         # market proxy รายวัน (log)
    for t in range(1, T):
        first_pass = log_p[t] - log_p[t - 1][match_slots(t, 0.0)]
        usable = np.abs(first_pass) < cap
        shift = np.median(first_pass[usable]) if usable.any() else 0.0
        prev_slot[t] = match_slots(t, shift)
        log_ret[t] = log_p[t] - log_p[t - 1][prev_slot[t]]
        usable = np.abs(log_ret[t] - shift) < cap
        mkt[t] = np.median(log_ret[t][usable]) if usable.any() else shift
    mkt_cum = np.cumsum(mkt)                  # market proxy สะสม — ใช้ปรับราคาเมื่อเทียบข้ามวัน

    # ---- ขั้น 3: real หรือ exit ----
    dormant = []                              # หุ้นที่เพิ่งออกจาก top 10: [log price วันสุดท้าย, วันสุดท้าย, ช่อง]

    def matches_dormant(t, slot):
        '''หลักฐาน (ก): ราคาตัวที่เข้ามาช่อง slot วัน t ตรงกับหุ้นที่เพิ่งออกไป -> คืน entry นั้น (ตัวแรกที่ตรง)'''
        for entry in dormant:
            log_price_exit, last_day, _ = entry
            if abs(log_p[t, slot] - (log_price_exit + mkt_cum[t] - mkt_cum[last_day])) < tol(t - last_day):
                return entry
        return None

    def exiting_company_returns(t, slot, prev):
        '''หลักฐาน (ข): หุ้นที่อยู่ช่อง prev เมื่อวาน กลับมาโผล่ที่ช่อง >= boundary ภายใน look_days วัน (พร้อมกระโดดเข้ามาใหม่)
        และตัวที่เข้ามาแทนในช่อง slot วัน t หายไปในวันนั้น -> คืนจำนวนวัน (None = ไม่พบ)'''
        for k in range(1, look_days + 1):
            if t + k >= T:
                break
            came_back = np.abs(log_p[t + k] - (log_p[t - 1, prev] + mkt_cum[t + k] - mkt_cum[t - 1])) < tol(k + 1)
            came_back[:boundary] = False
            jumped_in = np.abs((log_p[t + k] - log_p[t + k - 1]) - mkt[t + k]) > th_idio
            entrant_gone = not (np.abs(log_p[t + k] - (log_p[t, slot] + mkt_cum[t + k] - mkt_cum[t])) < tol(k)).any()
            if (came_back & jumped_in).any() and entrant_gone:
                return k
        return None

    ret = np.full((T, K), np.nan)             # log return ที่ยอมรับว่าเป็นบริษัทเดิม
    src = np.full((T, K), -1, int)            # ช่องเมื่อวานของค่าที่ยอมรับ (-1 = exit)
    events = []
    for t in range(1, T):
        exits_today = []
        for slot in range(K):
            prev = prev_slot[t, slot]
            abs_ret = abs(log_ret[t, slot])                     # ขนาดการขยับ
            abs_idio = abs(log_ret[t, slot] - mkt[t])           # ขนาดการขยับหลังหักตลาด (idiosyncratic)
            at_boundary = slot >= boundary or prev >= boundary
            status, reason = 'real', ''
            if abs_ret > cap:
                status, reason = 'exit', f'|log return| {abs_ret:.0%} > {cap:.0%}'
            elif at_boundary and abs_idio > th_big:
                status, reason = 'exit', f'idiosyncratic jump {abs_idio:.0%} > {th_big:.0%} at rank 9-10'
            elif at_boundary and abs_idio > th_idio:
                hit = matches_dormant(t, slot)
                if hit is not None:
                    status, reason = 'exit', f'entrant = company that left {t - hit[1]}d ago'
                    dormant.remove(hit)
                else:
                    k_back = exiting_company_returns(t, slot, prev)
                    if k_back is not None:
                        status, reason = 'exit', f'exiting company returns after {k_back}d'
            if status == 'exit':
                exits_today.append(prev)
            else:
                ret[t, slot] = log_ret[t, slot]
                src[t, slot] = prev
            if abs_idio > th_idio or abs_ret > cap:
                events.append({'date': prices.index[t], 'slot': slot + 1, 'prev_slot': prev + 1,
                               'prev_value': np.exp(log_p[t - 1, prev]), 'new_value': np.exp(log_p[t, slot]),
                               'pct_move': 100 * np.expm1(log_ret[t, slot]), 'top10_median_%': 100 * np.expm1(mkt[t]),
                               'status': status, 'reason': reason})
        for prev in exits_today:
            dormant.append([log_p[t - 1, prev], t - 1, prev])
        dormant[:] = [d for d in dormant if t - d[1] <= dormant_days]

    out = pd.DataFrame(np.expm1(ret), index=prices.index, columns=RET_COLS)
    matched = src >= 0
    out['swapped'] = ((src != np.arange(K)[None, :]) & matched).any(axis=1)
    out['n_unmatched'] = (~matched).sum(axis=1)
    out.iloc[0, out.columns.get_loc('n_unmatched')] = 0                    # แถวแรกไม่มีวันก่อนหน้า
    diag = pd.DataFrame(events, columns=['date', 'slot', 'prev_slot', 'prev_value', 'new_value', 'pct_move',
                                         'top10_median_%', 'status', 'reason']).set_index('date')
    market_proxy = pd.Series(np.expm1(mkt) * 100, index=prices.index, name='top10_median_%')
    return out, diag, market_proxy


# ----------------------------------------------------------------------------- dataset
def build_dataset(path: Path | str = CSV_PATH, impute: str | None = "cross_mean", percent: bool = True):
    """
    impute:
      "cross_mean" - เติม NaN ด้วยค่าเฉลี่ยของช่องอื่นในวันเดียวกัน (แนะนำ: ใช้กับ SVR/LSTM/linear ได้)
      "zero"       - เติม 0
      None         - ปล่อย NaN ไว้ (XGBoost/AutoGluon จัดการเองได้)
    percent: True = return ทุกคอลัมน์อยู่ในหน่วย % (เหมือน notebook)
    """
    prices = load_prices(path)
    ref = pd.DataFrame([REF_ROW]); ref["date"] = pd.to_datetime(ref["date"]); ref = ref.set_index("date")
    prices_ext = pd.concat([ref, prices])                # แถวอ้างอิง + ข้อมูล
    rets, diag, _ = unswap_returns(prices_ext)
    df = rets.iloc[1:].copy()                            # ตัดแถวอ้างอิงออก -> ทุกแถวมี return

    if impute == "cross_mean":
        row_mean = df[RET_COLS].mean(axis=1)
        for c in RET_COLS:
            df[c] = df[c].fillna(row_mean)
    elif impute == "zero":
        df[RET_COLS] = df[RET_COLS].fillna(0.0)
    elif impute is not None:
        raise ValueError(f"impute ไม่รู้จัก: {impute}")

    df[TARGET_RET] = prices_ext["target"].pct_change().iloc[1:]
    if percent:
        df[RET_COLS + [TARGET_RET]] *= 100
    df["target_level"] = prices["target"]
    return df, diag


# ----------------------------------------------------------------------------- split
def get_folds() -> list[dict]:
    return [
        {"name": f"fold{k}", "train": np.arange(0, tr_end), "val": np.arange(tr_end, va_end)}
        for k, (tr_end, va_end) in enumerate(FOLD_BOUNDS, start=1)
    ]


def get_test() -> dict:
    return {"name": "test", "train": np.arange(0, TEST_START), "val": np.arange(TEST_START, N_ROWS)}


def get_xy(df: pd.DataFrame, idx: np.ndarray, features: list[str] = RET_COLS, target: str = TARGET_RET):
    """ดึง X, y ตาม index ของ split; ตัดแถวที่ target เป็น NaN (แถวแรกสุด) ออก"""
    sub = df.iloc[idx].dropna(subset=[target])
    return sub[features], sub[target]


def scale_fold(X_train: pd.DataFrame, *others: pd.DataFrame, kind: str = "minmax"):
    """scaler fit บน train ของ fold เท่านั้น แล้ว transform ชุดอื่น — ใช้กับ SVR/LSTM/Ridge
    kind = "minmax" (x−min)/(max−min) ที่ CV เลือกใน notebook | "standard" z-score (x−μ)/σ"""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = (MinMaxScaler() if kind == "minmax" else StandardScaler()).fit(X_train)
    return (scaler, scaler.transform(X_train), *[scaler.transform(o) for o in others])


# ----------------------------------------------------------------------------- สรุป
def describe_splits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in get_folds() + [get_test()]:
        tr, va = df.iloc[f["train"]], df.iloc[f["val"]]
        rows.append({
            "split": f["name"],
            "train_rows": len(tr) - int(tr[TARGET_RET].isna().sum()),
            "train_from": tr.index[0].date(),
            "train_to": tr.index[-1].date(),
            "val_rows": len(va),
            "val_from": va.index[0].date(),
            "val_to": va.index[-1].date(),
            "val_target_min": round(va["target_level"].min()),
            "val_target_max": round(va["target_level"].max()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    df, diag = build_dataset()

    print(f"rows: {len(df)}   |  วันที่มีการสลับช่อง: {int(df['swapped'].sum())}   |  "
          f"ช่องที่เป็นหุ้นเข้า/ออก top 10: {int(df['n_unmatched'].sum())} ช่อง ใน {int((df['n_unmatched'] > 0).sum())} วัน")
    print()
    print("สถิติ return (%):")
    print(df[RET_COLS + [TARGET_RET]].describe().T[["mean", "std", "min", "max"]].round(2))
    print()
    print("split:")
    print(describe_splits(df).to_string(index=False))
    print()
    print("event ที่ตัดสินเป็นหุ้นเข้า/ออกด้วยหลักฐาน (ไม่ใช่เกณฑ์ 35%):")
    print(diag[(diag["status"] == "exit") & ~diag["reason"].str.startswith("|log")].round(2).to_string())
    print()
    print(f"การขยับใหญ่ (>8% เทียบ median top 10) ที่เก็บไว้เป็นของจริง: {int((diag['status'] == 'real').sum())} event")
