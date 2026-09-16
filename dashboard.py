"""
Dashboard: อิทธิพลของหุ้น Top 10 (market cap) ต่อ S&P 500

รัน:      streamlit run dashboard.py
ต้องมี:   dashboard_data.pkl ในโฟลเดอร์เดียวกัน (สร้างจาก section 16 ของ sp500_top10_influence.ipynb)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Top-10 vs S&P 500', page_icon='📈', layout='wide')

DATA_PATH = Path(__file__).with_name('dashboard_data.pkl')
METRIC_COLS = ['R2', 'MAE', 'MSE', 'RMSE', 'MAPE_%']
METRIC_LABEL = {'R2': 'R²', 'MAE': 'MAE (จุด)', 'MSE': 'MSE (จุด²)', 'RMSE': 'RMSE (จุด)', 'MAPE_%': 'MAPE (%)'}
HIGHER_BETTER = {'R2': True, 'MAE': False, 'MSE': False, 'RMSE': False, 'MAPE_%': False}
METRIC_FMT = {'R2': '{:.3f}', 'MAE': '{:.1f}', 'MSE': '{:.0f}', 'RMSE': '{:.1f}', 'MAPE_%': '{:.3f}'}
LEVEL_NOTE = 'ระดับที่ทำนาย = ระดับดัชนีเมื่อวาน × (1 + return ที่ทำนาย) แล้วเทียบกับระดับจริงของวันนั้น'
METRIC_HELP = {
    'R2': 'สัดส่วนความแปรปรวนของ return รายวันของ S&P ที่โมเดลอธิบายได้ (1 = อธิบายได้หมด) — ตัวเดียวที่คำนวณบน return',
    'MAE': 'ค่าเฉลี่ยของ |ระดับจริง − ระดับที่ทำนาย| หน่วยเป็นจุดดัชนี — ' + LEVEL_NOTE,
    'MSE': 'ค่าเฉลี่ยของ (ระดับจริง − ระดับที่ทำนาย)² หน่วยจุด² — ลงโทษวันที่พลาดมากเป็นพิเศษ',
    'RMSE': 'รากที่สองของ MSE กลับมาอยู่ในหน่วยจุดดัชนีเหมือน MAE',
    'MAPE_%': 'ค่าเฉลี่ยของ |ระดับจริง − ระดับที่ทำนาย| / ระดับจริง × 100 — อ่านว่า "ทำนายระดับดัชนีพลาดเฉลี่ยกี่ % ของระดับจริง"',
}
MAPE_EXPLAIN = '''
โมเดลทำนาย *return* รายวัน แต่ error ทั้ง 4 ตัว (MAE, MSE, RMSE, MAPE) วัดบน **ระดับดัชนี** เพื่อให้อ่านเป็นจุดดัชนีได้:
ระดับที่ทำนาย = ระดับเมื่อวาน × (1 + return ที่ทำนาย) แล้วเทียบกับระดับจริงของวันนั้น (R² ตัวเดียวที่วัดบน return)

- **MAE** = ค่าเฉลี่ยของ |ระดับจริง − ระดับที่ทำนาย| → "พลาดเฉลี่ยกี่จุด"
- **MSE** = ค่าเฉลี่ยของ (ระดับจริง − ระดับที่ทำนาย)² → ลงโทษวันที่พลาดมากเป็นพิเศษ (หน่วยจุด²)
- **RMSE** = √MSE → กลับมาเป็นจุด แต่ยังไวต่อวันที่พลาดมาก
- **MAPE** = ค่าเฉลี่ยของ |ระดับจริง − ระดับที่ทำนาย| ÷ ระดับจริง × 100 → "พลาดกี่ % ของค่าจริง"

ตัวอย่าง: ดัชนีจริง 5,000 จุด ทำนาย 5,015 จุด → พลาด 15 จุด (MAE) = 15 / 5,000 = **0.3%** (MAPE)

ทำไมไม่คำนวณ MAPE บน return ตรงๆ: วันที่ดัชนีแทบไม่ขยับ เช่น return จริง +0.01% ทำนาย +0.30% → |0.01 − 0.30| ÷ 0.01 = **2,900%**
ทั้งที่พลาดแค่ 0.29 % point — ตัวหารเล็กมากทำให้ตัวเลขไร้ความหมาย และวันแบบนี้มีเป็นสิบ % ของทั้งหมด ค่าเฉลี่ยจึงออกมาหลายร้อย %
'''
SPLITS = ['fold1', 'fold2', 'fold3', 'fold4', 'test']
SPLIT_NOTE = {'fold1': 'val 3/2018–8/2019', 'fold2': 'val 8/2019–2/2021 (COVID)', 'fold3': 'val 2/2021–7/2022 (bear 2022)',
              'fold4': 'val 7/2022–12/2023', 'test': 'test 12/2023–12/2025'}
RANKS = [f'rank {k}' for k in range(1, 11)]
# ----------------------------------------------------------------------------- data
@st.cache_resource(show_spinner='กำลังโหลดผลลัพธ์...')
def load_bundle(path: str, mtime: float):
    with open(path, 'rb') as f:
        return pickle.load(f)


def calc_metrics(y, p, prev_level, level):
    '''R2 บน return (%); MAE/MSE/RMSE (จุด) และ MAPE (%) บนระดับดัชนี (ระดับที่ทำนาย = ระดับเมื่อวาน × (1 + return ที่ทำนาย))'''
    y, p = np.asarray(y, float), np.asarray(p, float)
    level = np.asarray(level, float)
    pred_level = np.asarray(prev_level, float) * (1 + p / 100)
    return {'R2': float(r2_score(y, p)), 'MAE': float(mean_absolute_error(level, pred_level)), 'MSE': float(mean_squared_error(level, pred_level)),
            'RMSE': float(np.sqrt(mean_squared_error(level, pred_level))),
            'MAPE_%': float(np.mean(np.abs((pred_level - level) / level)) * 100)}


def csv_bytes(df: pd.DataFrame, index=True) -> bytes:
    return df.to_csv(index=index, float_format='%.6f').encode('utf-8-sig')      # BOM ให้ Excel เปิดภาษาไทย/ตัวเลขถูก


def build_datasets(B):
    '''train/test dataset ตามที่โมเดลใช้จริง (return %, un-swapped) + คอลัมน์ cv_role'''
    df = B['returns']; ret_cols = [f'r_{k}' for k in range(1, 11)]
    role = np.array(['always train (rows 1-551)'] * len(df), dtype=object)
    for name, (a, b) in zip(['fold1 val', 'fold2 val', 'fold3 val', 'fold4 val'], [(551, 916), (916, 1281), (1281, 1646), (1646, 2011)]):
        role[a:b] = name
    role[2011:] = 'test'
    out = df[ret_cols + ['r_target', 'target_level']].copy(); out.insert(0, 'cv_role', role)
    return out.iloc[:2011], out.iloc[2011:]


def download_section(B, key='dl'):
    st.subheader('ดาวน์โหลดข้อมูลและผลลัพธ์ (CSV)')
    train, test = build_datasets(B)
    res = B['results'].copy(); res['split'] = pd.Categorical(res['split'], SPLITS); res = res.sort_values(['model', 'split'])
    pred = B['predictions'].sort_values(['model', 'split', 'date'])
    items = [
        ('train_dataset.csv', 'Train set (2,011 แถว: feature r_1..r_10 + r_target %, cv_role)', csv_bytes(train)),
        ('test_dataset.csv', 'Test set (503 แถว)', csv_bytes(test)),
        ('results_summary.csv', 'Metric ทุกโมเดล ทุก split (R², MAE, MSE, RMSE, MAPE)', csv_bytes(res, index=False)),
        ('predictions.csv', 'ค่าทำนายรายวันของทุกโมเดล ทุก split (y_true, y_pred)', csv_bytes(pred, index=False)),
        ('returns_table.csv', 'ตาราง return หลัง un-swap ทั้ง 2,514 วัน + swapped / n_unmatched', csv_bytes(B['returns'])),
        ('unswap_diagnostics.csv', 'Event หุ้นเข้า/ออกและการขยับใหญ่ พร้อมเหตุผล', csv_bytes(B['diag'])),
        ('feature_importance.csv', 'Permutation importance บน test ทุกโมเดล', csv_bytes(B['importance'])),
        ('yearly_influence.csv', 'R² / beta / coefficient รายปี', csv_bytes(B['yearly'])),
    ]
    cols = st.columns(4)
    for i, (fname, desc, data) in enumerate(items):
        with cols[i % 4]:
            st.download_button(f'⬇ {fname}', data=data, file_name=fname, mime='text/csv', help=desc, key=f'{key}_{fname}', width='stretch')
            st.caption(desc)


def fmt_table(df: pd.DataFrame, cols=None):
    cols = cols or [c for c in METRIC_COLS if c in df.columns]
    fmt = {c: METRIC_FMT[c] for c in cols}
    sty = df.style.format(fmt)
    for c in cols:
        if HIGHER_BETTER[c]:
            sty = sty.highlight_max(subset=[c], color='#c8e6c9')
        else:
            sty = sty.highlight_min(subset=[c], color='#c8e6c9')
    return sty


def line_per_fold(res: pd.DataFrame, metric: str):
    d = res[res.split != 'test'].pivot(index='split', columns='model', values=metric).loc[SPLITS[:4]]
    fig = px.line(d, markers=True, labels={'value': METRIC_LABEL[metric], 'split': 'fold', 'model': 'โมเดล'})
    fig.update_layout(height=380, legend_title_text='', margin=dict(t=30, b=10))
    return fig


# ----------------------------------------------------------------------------- pages
def page_overview(B):
    meta, res = B['meta'], B['results']
    st.title('อิทธิพลของหุ้น Top 10 (market cap) ต่อ S&P 500')
    st.caption(f"ข้อมูล {meta['n_rows']:,} วันซื้อขาย ({meta['period']}) · same-day return ของอันดับ 1–10 → return ดัชนี · "
               f"train 2,011 / test 503 · expanding CV 4 fold × 365 วัน · รันเมื่อ {meta['run_at']}")

    test = res[res.split == 'test'].set_index('model').sort_values('R2', ascending=False)
    best = test.index[0]
    c = st.columns(6)
    c[0].metric('โมเดลที่ดีที่สุด (test)', best)
    c[1].metric('R² test', f"{test.loc[best, 'R2']:.3f}", help=METRIC_HELP['R2'])
    c[2].metric('MAE test', f"{test.loc[best, 'MAE']:.1f} จุด", help=METRIC_HELP['MAE'])
    c[3].metric('MSE test', f"{test.loc[best, 'MSE']:.0f} จุด²", help=METRIC_HELP['MSE'])
    c[4].metric('RMSE test', f"{test.loc[best, 'RMSE']:.1f} จุด", help=METRIC_HELP['RMSE'])
    c[5].metric('MAPE test', f"{test.loc[best, 'MAPE_%']:.3f} %", help=METRIC_HELP['MAPE_%'])
    with st.expander('MAE / MSE / RMSE / MAPE คำนวณยังไงในงานนี้'):
        st.markdown(MAPE_EXPLAIN)
        lv = B['returns']['target_level']; te = B['predictions'].query("split == 'test'").drop_duplicates('date').sort_values('date')
        pos = pd.Series(np.arange(len(lv)), index=lv.index).loc[te.date].to_numpy()
        diff = lv.to_numpy()[pos] - lv.to_numpy()[pos - 1]
        st.caption(f'ไว้เทียบ: ถ้าทำนายว่า "ดัชนีไม่เปลี่ยนจากเมื่อวาน" ทุกวัน จะได้ MAE = {np.mean(np.abs(diff)):.1f} จุด, RMSE = {np.sqrt(np.mean(diff ** 2)):.1f} จุด, '
                   f'MAPE = {np.mean(np.abs(te.y_true)):.3f}% (= ขนาดการขยับเฉลี่ยของดัชนี) → โมเดลที่ดีที่สุดลด error ลงเหลือ {test.loc[best, "MAE"] / np.mean(np.abs(diff)) * 100:.0f}% ของ baseline')

    st.markdown('---')
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader('ตารางเปรียบเทียบโมเดล')
        choice = st.selectbox('ช่วง', ['test', 'CV mean (fold1–4)'] + SPLITS[:4], format_func=lambda s: f'{s} — {SPLIT_NOTE[s]}' if s in SPLIT_NOTE else s)
        if choice.startswith('CV'):
            tbl = res[res.split != 'test'].groupby('model')[METRIC_COLS].mean()
        else:
            tbl = res[res.split == choice].set_index('model')[METRIC_COLS]
        tbl = tbl.sort_values('R2', ascending=False)
        st.dataframe(fmt_table(tbl), width='stretch')
        st.caption('สีเขียว = ดีที่สุดในคอลัมน์นั้น · R² คำนวณบน return รายวัน · MAE/MSE/RMSE (จุดดัชนี) และ MAPE (%) คำนวณบนระดับดัชนี (ดูคำอธิบายด้านบน)')
    with right:
        st.subheader('เทียบตาม metric')
        metric = st.selectbox('metric', METRIC_COLS, format_func=lambda m: METRIC_LABEL[m], key='ov_metric')
        d = tbl[metric].sort_values(ascending=not HIGHER_BETTER[metric])
        fig = px.bar(d, orientation='h', labels={'value': METRIC_LABEL[metric], 'model': ''}, text=[METRIC_FMT[metric].format(v) for v in d])
        fig.update_layout(height=360, showlegend=False, margin=dict(t=20, b=10), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, width='stretch')

    st.markdown('---')
    st.subheader('ข้อค้นพบหลัก')
    yearly = B['yearly']
    k1, k2, k3 = st.columns(3)
    k1.info(f"**Top 10 อธิบาย return รายวันของ S&P ได้ ~{test['R2'].max() * 100:.0f}%** (test) · CV เฉลี่ยของโมเดล tabular "
            f"{res[(res.split != 'test') & (res.model != 'LSTM')].groupby('model')['R2'].mean().min():.2f}–"
            f"{res[(res.split != 'test') & (res.model != 'LSTM')].groupby('model')['R2'].mean().max():.2f} · "
            f"R² รายปีสูงสุดในปีวิกฤต ({yearly['R2_10slots'].idxmax()}: {yearly['R2_10slots'].max():.2f})")
    k2.info(f"**Sensitivity ขึ้นกับ regime** — ดัชนีขยับ {yearly['beta_EW'].max():.2f}% ต่อ 1% ของ top 10 ในปี {yearly['beta_EW'].idxmax()} "
            f"แต่แค่ {yearly['beta_EW'].min():.2f}% ในปี {yearly['beta_EW'].idxmin()} — mega-cap ขยับด้วยเรื่องตัวเองมากขึ้น ตลาดตามน้อยลง")
    k3.info('**Importance กระจุกที่อันดับ 7–10 ไม่ใช่ 1–3** — เพราะอันดับ 1–5 correlate กันสูงจึงแชร์เครดิต และอันดับ 7–10 เป็น proxy '
            'ของหุ้นอีก 490 ตัว (ดูหน้า Feature importance) — ไม่ใช่น้ำหนัก market cap')
    st.markdown('---')
    download_section(B, key='ov')


def page_compare(B):
    res, pred = B['results'], B['predictions']
    st.title('เปรียบเทียบโมเดล')
    metric = st.selectbox('metric', METRIC_COLS, format_func=lambda m: METRIC_LABEL[m])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader(f'{METRIC_LABEL[metric]} ราย fold (validation)')
        st.plotly_chart(line_per_fold(res, metric), width='stretch')
        st.caption(' · '.join(f'{k}: {v}' for k, v in SPLIT_NOTE.items() if k != 'test'))
    with c2:
        st.subheader(f'{METRIC_LABEL[metric]} บน test')
        d = res[res.split == 'test'].set_index('model')[metric].sort_values(ascending=not HIGHER_BETTER[metric])
        fig = px.bar(d, orientation='h', text=[METRIC_FMT[metric].format(v) for v in d], labels={'value': METRIC_LABEL[metric], 'model': ''})
        fig.update_layout(height=380, showlegend=False, margin=dict(t=20, b=10), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, width='stretch')

    if B.get('significance') is not None:
        st.subheader('ความต่างบน test มีนัยสำคัญไหม? (moving-block bootstrap 2,000 รอบ + Diebold–Mariano)')
        sg = B['significance']
        st.dataframe(sg.style.format({'R2 test': '{:.3f}', 'DM stat': '{:.2f}', 'DM p-value': '{:.3f}'}, na_rep=''), width='stretch')
        st.caption('R2 95% CI = ช่วงความเชื่อมั่นของ R² บน test จาก block bootstrap (block 21 วัน) · dR2 vs Linear = ความต่างของ R² จาก Linear ใน bootstrap เดียวกัน '
                   '(CI ไม่คร่อม 0 = ต่างอย่างมีนัยสำคัญ) · DM = Diebold–Mariano test บน squared error (Newey-West lag 5); DM stat > 0 และ p < 0.05 = ดีกว่า Linear อย่างมีนัยสำคัญ')
    st.markdown('---')
    st.subheader('Actual vs Predicted')
    a, b = st.columns(2)
    model = a.selectbox('โมเดล', list(dict.fromkeys(pred.model)))
    split = b.selectbox('ช่วง', SPLITS, index=4, format_func=lambda s: f'{s} — {SPLIT_NOTE[s]}')
    d = pred[(pred.model == model) & (pred.split == split)].sort_values('date')
    lvl = B['returns']['target_level']
    pos = pd.Series(np.arange(len(lvl)), index=lvl.index).loc[d.date].to_numpy()
    m = calc_metrics(d.y_true, d.y_pred, prev_level=lvl.to_numpy()[pos - 1], level=lvl.to_numpy()[pos])
    cc = st.columns(5)
    for i, k in enumerate(METRIC_COLS):
        cc[i].metric(METRIC_LABEL[k], METRIC_FMT[k].format(m[k]), help=METRIC_HELP.get(k))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.date, y=d.y_true, name='S&P 500 actual (%)', line=dict(width=1.2, color='#333')))
    fig.add_trace(go.Scatter(x=d.date, y=d.y_pred, name=f'{model} predicted (%)', line=dict(width=1.2, color='#e4572e'), opacity=0.85))
    fig.update_layout(height=380, margin=dict(t=30, b=10), legend=dict(orientation='h', y=1.08), yaxis_title='return รายวัน (%)')
    st.plotly_chart(fig, width='stretch')

    s1, s2 = st.columns(2)
    with s1:
        lim = float(max(abs(d.y_true).max(), abs(d.y_pred).max()) * 1.05)
        fig = px.scatter(d, x='y_true', y='y_pred', hover_data=['date'], labels={'y_true': 'actual (%)', 'y_pred': 'predicted (%)'}, opacity=0.6)
        fig.add_shape(type='line', x0=-lim, y0=-lim, x1=lim, y1=lim, line=dict(dash='dash', color='gray'))
        fig.update_layout(height=380, title='scatter: actual vs predicted (เส้นประ = ทำนายถูกเป๊ะ)', margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')
    with s2:
        resid = d.y_pred - d.y_true
        fig = px.histogram(resid, nbins=60, labels={'value': 'residual = predicted − actual (%)'})
        fig.update_layout(height=380, title=f'residual — mean {resid.mean():+.3f}, std {resid.std():.3f}', showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')


def page_importance(B):
    imp, coef_tbl, struct = B['importance'], B['coef_tbl'], B['struct']
    st.title('Feature importance — อันดับไหนสำคัญ')
    st.info('**วิธีอ่าน:** importance/coefficient ไม่เรียงตามอันดับ market cap (อันดับ 7–10 มักสูงกว่า 1–5) — ไม่ใช่เพราะอันดับ 10 มีน้ำหนักในดัชนี'
            'มากกว่าอันดับ 1 แต่เพราะ (ก) อันดับ 1–5 เป็นกลุ่มเทคที่ correlate กันสูง จึงแชร์เครดิตกันใน multiple regression และ (ข) หุ้นที่อันดับ 7–10 '
            '(การเงิน สุขภาพ พลังงาน) ขยับคล้ายหุ้นอีก 490 ตัว จึงเป็น proxy ของส่วนที่เหลือของดัชนี → ตัวเลขนี้ตอบว่า *ช่องไหนให้ข้อมูลที่ช่องอื่นไม่มี* '
            'ไม่ใช่ *ช่องไหนขยับดัชนีมากที่สุดเชิงกลไก* (อย่างหลังต้องใช้ market cap × return)')

    st.subheader('Permutation importance บน test (R² ที่ลดลงเมื่อสุ่มสลับ return ของอันดับนั้น)')
    models = st.multiselect('โมเดล', list(imp.columns), default=list(imp.columns))
    if models:
        d = imp[models].reset_index().melt(id_vars='index', var_name='โมเดล', value_name='R2 drop').rename(columns={'index': 'rank'})
        fig = px.bar(d, x='rank', y='R2 drop', color='โมเดล', barmode='group')
        fig.update_layout(height=400, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption('ค่าติดลบ = สลับ feature แล้ว R² บน test กลับดีขึ้น → coefficient ของอันดับนั้นที่ fit จาก 2016–2023 ใช้กับ 2024–25 ไม่ได้ (regime เปลี่ยน)')

    if B.get('importance_order') is not None:
        st.subheader('อันดับไหนส่งอิทธิพลต่อดัชนี — แยกตามโมเดล')
        t1, t2 = st.columns(2)
        with t1:
            st.markdown('**ลำดับอิทธิพล** (1 = อันดับที่โมเดลนั้นพึ่งพามากที่สุด · consensus = ค่าเฉลี่ยลำดับข้ามโมเดล)')
            io = B['importance_order'].sort_values('consensus')
            st.dataframe(io.style.format({'consensus': '{:.1f}'}).background_gradient(subset=[c for c in io.columns if c != 'consensus'], cmap='Greens_r', vmin=1, vmax=10), width='stretch')
        with t2:
            st.markdown('**สัดส่วน importance** ของแต่ละอันดับต่อ importance รวมของโมเดล (%, นับเฉพาะค่าบวก)')
            st.dataframe(B['importance_share'].style.format('{:.1f}').background_gradient(cmap='Greens', axis=None), width='stretch')
        st.caption('ทุกโมเดลเห็นตรงกันว่าอันดับ 8–10 (และอันดับ 1) ให้ข้อมูลที่ช่องอื่นไม่มี ส่วนอันดับ 4–6 แทบไม่เพิ่มข้อมูลเมื่อมีอันดับอื่นอยู่แล้ว — '
                   'อ่านคู่กับกล่องด้านบน: นี่คือ "ช่องไหนให้ข้อมูลเฉพาะตัว" ไม่ใช่น้ำหนัก market cap')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('SHAP — XGBoost (mean |SHAP| บน test)')
        sm = B.get('shap_mean')
        if sm is not None:
            sm = sm.reindex(RANKS)
            fig = px.bar(sm, labels={'value': 'mean |SHAP| (%)', 'index': ''})
            fig.update_layout(height=360, showlegend=False, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width='stretch')
        else:
            st.write('ไม่มีผล SHAP ในการรันนี้')
    with c2:
        st.subheader('Coefficient ของ Linear ราย fold')
        ct = coef_tbl.drop(index='sum', errors='ignore')
        ct.index = RANKS[:len(ct)]
        fig = px.imshow(ct, text_auto='.3f', color_continuous_scale='Blues', aspect='auto', labels={'x': 'fit บน train ของ', 'y': ''})
        fig.update_layout(height=360, margin=dict(t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')
        st.caption('ผลรวม coefficient: ' + ' · '.join(f'{c}: {v:.2f}' for c, v in coef_tbl.loc['sum'].items()) + '  (sensitivity รวม — ไม่ใช่น้ำหนัก market cap)')

    if B.get('shap_slopes') is not None:
        st.subheader('SHAP dependence: ความชันช่วงกลาง vs ช่วงปลาย (non-linearity)')
        ss = B['shap_slopes'].copy(); ss.index = [f'rank {k}' for k in ss.index]
        st.dataframe(ss.T.style.format('{:.3f}', na_rep=''), width='stretch')
        st.caption('slope = SHAP (% point ของดัชนี) ต่อ return 1% ของอันดับนั้น ในช่วง |return| < 2% เทียบกับ |return| > 4% — ratio < 1 = แบนที่ปลาย: '
                   'การขยับใหญ่ของหุ้นตัวเดียวส่งผลต่อดัชนีน้อยกว่าสัดส่วนเชิงเส้น (ชัดที่อันดับ 5–10) ซึ่งเป็นสิ่งที่ tree model จับได้แต่ linear ทำไม่ได้')

    st.subheader('ทำไมอันดับ 7–10 ถึงเด่น — marginal vs partial (train 2016–2023)')
    s2 = struct.copy()
    if B.get('vif') is not None:
        s2['VIF'] = B['vif'].to_numpy()
    st.dataframe(s2.style.format('{:.3f}').format({'VIF': '{:.2f}'}), width='stretch')
    st.caption('corr_with_SP = correlation ตรงๆ กับ S&P (อันดับ 1–4 สูงสุด) · OLS coef = partial effect เมื่อคุมอันดับอื่น (อันดับ 9–10 สูงสุด) · '
               'corr_with_other9 = correlate กับกลุ่มแค่ไหน (อันดับ 1–5 สูง → แชร์เครดิต) · VIF 1–5 = collinearity ปานกลาง (>10 = รุนแรง)')
    with st.expander('OLS summary (HAC standard errors)'):
        st.code(B['ols_summary'])


def page_time(B):
    yearly, roll = B['yearly'], B['rolling_r2']
    st.title('อิทธิพลเปลี่ยนตามเวลา')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(yearly['R2_10slots'], markers=True, labels={'value': 'R²', 'year': 'ปี'})
        fig.update_layout(height=360, title='R² รายปี: return รายวันของ S&P ที่อธิบายได้ด้วย top 10 (linear, in-sample)', showlegend=False, yaxis_range=[0, 1], margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.line(yearly[['beta_EW', 'beta_EW_rank1to8']], markers=True, labels={'value': 'beta', 'year': 'ปี'})
        fig.update_layout(height=360, title='Sensitivity: ดัชนีขยับกี่ % ต่อ 1% ของค่าเฉลี่ย top 10', yaxis_range=[0, 1.1], legend_title_text='', margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')
    st.caption('beta สูงในปีที่ตลาดผันผวน (2018, 2020, 2022) และต่ำในปีที่ตลาดนิ่ง (2017, 2021, 2023, 2024) — ปี 2023–24 ต่ำสุดในรอบ 10 ปี '
               'แม้จะตัดอันดับ 9–10 ที่เป็น boundary ออก (beta_EW_rank1to8) ก็ยังเห็นภาพเดียวกัน')
    st.dataframe(yearly[['R2_10slots', 'corr_EW', 'beta_EW', 'beta_EW_rank1to8', 'sum_coef']].style.format('{:.3f}'), width='stretch')

    if B.get('corr_period') is not None:
        st.subheader('correlation ภายในกลุ่ม top 10 แยกช่วง')
        st.dataframe(B['corr_period'].style.format('{:.2f}'), width='stretch')
        st.caption('corr(rank, other 9) = correlation ของอันดับนั้นกับค่าเฉลี่ยของอีก 9 อันดับ — ช่วง 2023–25 อันดับ 1–4 ลดลงเหลือ 0.60–0.74 จาก 0.74–0.85 ในช่วง 2016–22 '
                   '→ mega-cap ขยับด้วยเรื่องของตัวเองมากขึ้น (AI) เป็นที่มาของ sensitivity ที่ลดลง')

    fig = px.line(roll, labels={'value': 'R²', 'index': ''})
    fig.update_layout(height=340, title='Rolling 252-day R² (linear, in-window)', showlegend=False, yaxis_range=[0, 1], margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch')


def page_data(B):
    meta, diag, df, chk, sc = B['meta'], B['diag'], B['returns'], B['unswap_check'], B['unswap_scatter']
    st.title('ข้อมูลและการ un-swap')
    c = st.columns(5)
    c[0].metric('วันซื้อขาย', f"{meta['n_rows']:,}")
    c[1].metric('วันที่มีการสลับอันดับ', f"{meta['swap_days']:,}")
    c[2].metric('วันที่มีหุ้นเข้า/ออก top 10', f"{meta['entry_exit_days']:,}")
    c[3].metric('corr(EW top-10, S&P)', f"{chk['corr_EW_SP']:.3f}")
    c[4].metric('corr คู่ราคาใกล้กัน (GOOGL/GOOG)', f"{chk['twin_corr']:.3f}", help=f"{chk['twin_pairs']} pair-days ที่ราคาต่างกัน <1% สองวันติด — ถ้าจับคู่ผิด ค่านี้จะต่ำ")
    if B.get('reference_row') is not None:
        rr = B['reference_row']
        st.caption(f"แถวแรกของข้อมูล (4/1/2016) มี return ด้วย โดยใช้แถวอ้างอิง {rr['date']} จาก Yahoo Finance (adj close ของ GOOGL, GOOG, MSFT, AAPL, AMZN, XOM, BRK-B, META, JNJ, VZ "
                   f"ตามช่อง 1–10 และ S&P 500 = {rr['target']:,.2f}) — ใช้คำนวณ return ของแถวแรกเท่านั้น ไม่นับเป็นข้อมูล ทำให้ train มี 551 / 2,011 แถวตรงตามแผน")

    st.subheader('ทำไมต้อง un-swap — Linear regression เดียวกัน ก่อน/หลัง')
    a, b = st.columns([1, 2])
    with a:
        t = pd.DataFrame({'train R²': [chk['naive_train_R2'], chk['unswapped_train_R2']], 'test R²': [chk['naive_test_R2'], chk['unswapped_test_R2']]},
                         index=['ใช้ return ของช่องเดิมตรงๆ', 'หลัง un-swap'])
        st.dataframe(t.style.format('{:.3f}'), width='stretch')
        st.caption('คอลัมน์ close_k เป็น "ช่องอันดับ" ไม่ใช่หุ้นตัวเดิม → return ของช่องเดิมมีค่ากระโดดปลอมๆ ทุกครั้งที่อันดับสลับ')
    with b:
        fig = px.scatter(sc.reset_index().rename(columns={'index': 'date'}).melt(id_vars=['date', 'sp'], var_name='แบบ', value_name='ew'),
                         x='ew', y='sp', facet_col='แบบ', opacity=0.35, labels={'ew': 'ค่าเฉลี่ย return top 10 (%)', 'sp': 'S&P return (%)'},
                         category_orders={'แบบ': ['ew_before', 'ew_after']})
        fig.for_each_annotation(lambda t_: t_.update(text={'แบบ=ew_before': 'ก่อน un-swap', 'แบบ=ew_after': 'หลัง un-swap'}.get(t_.text, t_.text)))
        fig.update_xaxes(range=[-15, 15])
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')
        n_out = int((sc['ew_before'].abs() > 15).sum())
        st.caption(f'สเกลเดียวกันทั้งสองรูป (±15%) — กราฟซ้ายมีอีก {n_out} วันที่ค่าเฉลี่ยกระโดดเกิน ±15% (สูงสุด {sc["ew_before"].max():+.0f}%) '
                   'ซึ่งเป็นค่าปลอมจากการสลับช่อง ไม่ได้แสดงในกรอบ')

    if B.get('nextday_r2') is not None:
        nd = B['nextday_r2']
        st.info(f"**Same-day ไม่ใช่การทำนายอนาคต:** ถ้าใช้ return ของ top 10 *เมื่อวาน* ทำนายดัชนี*วันนี้* Linear ได้ R² train {nd['train']:.3f} / test {nd['test']:.3f} (≈ 0) "
                f"เทียบ same-day {chk['unswapped_train_R2']:.2f} / {chk['unswapped_test_R2']:.2f} — ความสัมพันธ์ที่วัดเป็นเรื่องวันเดียวกัน (อิทธิพล) ไม่มี look-ahead")

    if B.get('sensitivity') is not None:
        st.subheader('Sensitivity: พารามิเตอร์ของ un-swap และวิธี impute')
        st.dataframe(B['sensitivity'].style.format({'entry/exit slots': '{:.0f}', 'twin corr': '{:.3f}', 'corr(EW,S&P)': '{:.3f}', 'linear R2 train': '{:.3f}', 'linear R2 test': '{:.3f}'}, na_rep=''), width='stretch')
        st.caption('ทุกแถว = un-swap ด้วยค่านั้น → impute → Linear (train 2016–23) → R² test 2024–25 · ผลระดับรวม (corr, R² train) แทบไม่เปลี่ยน · '
                   'rank penalty จำเป็นต่อความถูกต้องรายช่อง (twin corr) · กฎ boundary jump >20% มีผลต่อ R² test เพราะ event ปลอมจากการสลับ TSLA/AVGO เกือบทั้งหมดอยู่ในปี 2024 — '
                   'เกณฑ์ตั้งจากการตรวจ event กับราคาจริงของหุ้น (data cleaning) และแสดงผลเมื่อใช้เกณฑ์อื่น/ปิดกฎไว้ตรงนี้เพื่อความโปร่งใส')

    st.subheader('Dataset ที่โมเดลใช้จริง (หลัง un-swap, หน่วย %)')
    train, test = build_datasets(B)
    which = st.radio('ชุด', ['train (2,011 แถว)', 'test (503 แถว)'], horizontal=True, label_visibility='collapsed')
    show = train if which.startswith('train') else test
    st.dataframe(show.style.format({c: '{:.3f}' for c in show.columns if c.startswith('r_')} | {'target_level': '{:.2f}'}), width='stretch', height=320)
    st.caption('feature = r_1..r_10 (return รายวันของอันดับ 1–10) · target = r_target (return รายวันของ S&P 500 วันเดียวกัน) · target_level ไว้แปลง error เป็นจุด · '
               'cv_role = แถวนี้เป็น validation ของ fold ไหน (แถว 1–551 เป็น train ทุก fold)')
    download_section(B, key='data')

    st.subheader('หุ้นเข้า/ออก top 10 ที่ตัดสินด้วยหลักฐาน (ไม่ใช่เกณฑ์ 35%)')
    dfmt = {'prev_value': '{:.2f}', 'new_value': '{:.2f}', 'pct_move': '{:+.1f}', 'top10_median_%': '{:+.2f}'}
    diag = diag.copy(); diag.index = pd.to_datetime(diag.index).strftime('%Y-%m-%d')
    ex = diag[(diag.status == 'exit') & ~diag.reason.str.startswith('|log')].drop(columns='status')
    st.dataframe(ex.style.format(dfmt), width='stretch', height=300)
    st.subheader('การขยับใหญ่ (>8% เทียบ median ของ top 10) ที่เก็บไว้เป็นของจริง')
    real = diag[diag.status == 'real'].drop(columns=['status', 'reason'])
    st.dataframe(real.style.format(dfmt), width='stretch', height=300)
    st.caption('อันดับ 1–8: เก็บทุกการขยับที่ไม่เกิน 35% (เช่น META −19% 7/2018, NVDA +24% 5/2023, TSLA −21% 9/2020) · อันดับ 9–10: >20% ถือว่าเปลี่ยนตัว')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('หุ้นเข้า/ออก แยกตามอันดับ')
        ebr = B['exit_by_rank'].copy(); ebr.index = RANKS
        fig = px.bar(ebr, labels={'value': 'จำนวนวัน', 'index': ''}); fig.update_layout(height=300, showlegend=False, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader('สถิติ return รายวัน (%)')
        cols = [f'r_{k}' for k in range(1, 11)] + ['r_target']
        st.dataframe(df[cols].describe().T[['mean', 'std', 'min', 'max']].style.format('{:.2f}'), width='stretch', height=300)

    st.subheader('ทำไมไม่ใช้ระดับราคา (level)')
    ld, ls = B['level_demo'], B['level_series']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('placebo: ใช้แค่ "ลำดับวัน" ทำนาย level', f"train R² {ld['placebo_time_index_train_R2']:.3f}")
    m2.metric('Linear บน level', f"train {ld['Linear_level_train_R2']:.3f} / test {ld['Linear_level_test_R2']:.3f}")
    m3.metric('XGBoost บน level', f"train {ld['XGBoost_level_train_R2']:.3f} / test {ld['XGBoost_level_test_R2']:.3f}")
    if B.get('adf') is not None:
        m4.metric('ADF p-value  level / return', f"{B['adf']['level_p']:.3f} / {B['adf']['return_p']:.0e}",
                  help='Augmented Dickey-Fuller: p สูง = non-stationary (มี unit root) → regression บน level เป็น spurious; return รายวัน p ≈ 0 = stationary')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ls.index, y=ls['actual'], name='S&P 500', line=dict(color='#333', width=1.2)))
    for k, col in [('XGBoost', '#e4572e'), ('Linear', '#1f77b4')]:
        if k in ls:
            seg = ls[k].iloc[2011:]
            fig.add_trace(go.Scatter(x=seg.index, y=seg, name=f'{k} on levels (test)', line=dict(color=col, width=1.2)))
    fig.update_layout(height=360, title='โมเดล level: train R² ~0.98–1.00 แต่ XGBoost ทำนายเกินช่วงที่เคยเห็นไม่ได้ (test)', margin=dict(t=40, b=10), legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, width='stretch')


def page_search(B):
    st.title('Hyperparameter search (คะแนน = ค่าเฉลี่ย R² ของ 4 fold)')
    st.json(B['best_params'], expanded=False)
    st.caption(B.get('scaler_rule', ''))
    names = {'linear': 'Linear — scaler (none / z-score / min-max)', 'ridge': 'Ridge — scaler × alpha', 'xgboost': 'XGBoost — random search 40 configs × 3 scalers (120 การประเมิน)',
             'svr': 'SVR — grid scaler × C × epsilon × gamma (576 configs)', 'lstm': 'LSTM — window × hidden × scaler', 'autogluon': 'AutoGluon — scaler'}
    for key, title in names.items():
        t = B['search'].get(key)
        if t is None:
            continue
        st.subheader(title)
        st.dataframe(t.head(15).style.format(precision=4), width='stretch')
    if B.get('svr_kernels') is not None:
        st.subheader('SVR: ทำไมใช้ RBF kernel เดียว — เทียบ kernel อื่นด้วย protocol เดียวกัน')
        sk = B['svr_kernels'].copy(); sk['best params'] = sk['best params'].astype(str)
        st.dataframe(sk.style.format({'cv mean R2': '{:.3f}', 'cv min R2': '{:.3f}', 'test R2': '{:.3f}'}), width='stretch')
        st.dataframe(B['svr_stress'].style.format('{:+.2f}').background_gradient(cmap='RdBu', axis=None, vmin=-16, vmax=16), width='stretch')
        sk_cv, sk_te = B['svr_kernels']['cv mean R2'], B['svr_kernels']['test R2']
        st.caption(f'RBF ถูกกำหนดไว้ล่วงหน้า (kernel มาตรฐานที่ bounded) — ทุก kernel ใช้ scaler เดียวกับที่ CV เลือกให้ RBF ({B["best_params"]["svr"].get("scaler", "standard")}) · '
                   f'CV mean ของทุก kernel ต่างกันแค่ {sk_cv.max() - sk_cv.min():.3f} จึงแยกไม่ออกด้วย CV (kernel ที่ CV สูงสุด: {sk_cv.idxmax()}) · '
                   f'บน test อยู่ในช่วง {sk_te.min():.3f}–{sk_te.max():.3f} ซึ่งแคบกว่า CI ของ R² test (หน้า *ผลลัพธ์*) มาก · '
                   'ตาราง stress test = ค่าทำนายเมื่อหุ้นทั้ง 10 อันดับขยับ −12/−6/+6/+10% พร้อมกัน — เมื่อ input ถูกบีบลง [0,1] ด้วย min-max ไม่มี kernel ไหนระเบิด · '
                   'γ ที่ CV เลือกให้ RBF เล็กมาก (0.0005 บน input [0,1]) kernel จึงทำงานในโหมดเกือบเชิงเส้น (CV เท่ากับ linear kernel) · คอลัมน์ test ไม่ได้ใช้เลือก แสดงเพื่อความโปร่งใส')

    if B.get('scale_check') is not None:
        st.subheader('โมเดลไหนต้อง normalize และแบบไหน? — ทุกโมเดล search none / z-score / min-max ด้วย CV (scaler fit บน train ของ fold)')
        st.dataframe(B['scale_check'].style.format({'cv mean R2': '{:.4f}', 'test R2': '{:.4f}'}), width='stretch')
        sc = B['scale_check']['cv mean R2'].groupby(level='model', sort=False).agg(lambda x: x.max() - x.min())
        chosen = {k: v.get('scaler', 'none') for k, v in B['best_params'].items()}
        st.caption(f"scaler ที่ CV เลือก: {chosen} · ความต่างของ CV mean ระหว่าง scaler ต่อโมเดล: {', '.join(f'{k} {v:.4f}' for k, v in sc.items())} · "
                   f"กติกาเดียวกันทุกโมเดล: CV สูงสุด แต่ถ้า none ตามหลังไม่เกิน 0.001 ให้ none (parsimony) → เลือก none (scale ไม่ช่วย): {[k for k, v in chosen.items() if v == 'none']} | เลือก scale (ช่วยจริง): {[k for k, v in chosen.items() if v != 'none']} · "
                   'เหตุผลเชิงกลไก: OLS scale-equivariant · tree แบ่ง node ด้วยลำดับของค่า · AutoGluon มี preprocessing ของตัวเองต่อโมเดล · Ridge (penalty ขึ้นกับหน่วย) / SVR (RBF ใช้ระยะทาง) / LSTM (gate อิ่มตัว) ไวต่อ scale — '
                   'return มีหางหนา (−21% ถึง +24%) z-score ทำให้วันสุดขั้วมี |z| 5–10 ส่วน min-max บีบทุกค่าลง [0,1] · target ไม่ scale (หน่วย %)')

    if B.get('ag_leaderboard') is not None:
        st.subheader('AutoGluon leaderboard (final model, tuning_data = 20% ท้ายของ train)')
        st.dataframe(B['ag_leaderboard'][['model', 'score_val', 'fit_time', 'stack_level']].style.format({'score_val': '{:.4f}', 'fit_time': '{:.1f}'}), width='stretch')
    st.caption('ทุกการเลือก hyperparameter ใช้ CV เท่านั้น test ถูกวัดครั้งเดียวหลังเลือกแล้ว')


def page_method(B):
    st.title('วิธีการและข้อสรุป')
    st.markdown('''
### วิธีการ
1. **Return ไม่ใช่ level** — ราคา normalize แล้วยังมี trend ร่วม ทำให้ R² 0.99 แบบไม่มีความหมาย และ tree model ทำนายเกินช่วง train ไม่ได้ (หน้า *ข้อมูล*)
2. **Un-swap** — คอลัมน์เป็นช่องอันดับ ต้องจับคู่ราคาวัน t กับ t−1 ให้เป็นบริษัทเดิมก่อนคำนวณ return: Hungarian assignment + penalty การเลื่อนอันดับ + market proxy (median ของ 10 ช่อง) + กฎหุ้นเข้า/ออกเฉพาะอันดับ 9–10 ที่ต้องมีหลักฐาน; ช่องที่เปลี่ยนตัวเติมด้วยค่าเฉลี่ยของช่องอื่น
3. **Same-day** — return ของอันดับ 1–10 วัน t อธิบาย return ดัชนีวัน t (วัดอิทธิพล ไม่ใช่ทำนายอนาคต)
4. **Split** — train 2,011 วัน / test 503 วัน (12/2023–12/2025); expanding-window CV 4 fold × 365 วันใน train; scaler fit ต่อ fold; เลือก hyperparameter จาก CV เท่านั้น; test วัดครั้งเดียว
5. **โมเดล** — Linear, Ridge, XGBoost (random search), SVR (grid), LSTM (PyTorch, early stopping บน inner holdout), AutoGluon (tuning_data = 20% ท้ายของ train); *ทุกโมเดล* มีชนิด scaler (none / z-score / min-max) เป็น hyperparameter ที่เลือกจาก CV ด้วยกติกาเดียวกัน (CV สูงสุด; ถ้า none ตามหลังไม่เกิน 0.001 เลือก none)
6. **Metric** — R² คำนวณบน return รายวัน; MAE, MSE, RMSE (หน่วยจุดดัชนี) และ MAPE (%) คำนวณบนระดับดัชนี (ระดับที่ทำนาย = ระดับเมื่อวาน × (1 + return ที่ทำนาย)) — MAPE ไม่คำนวณบน return เพราะหารด้วยค่าจริงที่ใกล้ 0 จะได้ตัวเลขหลายร้อย % ที่ไม่สื่ออะไร (ดูคำอธิบายในหน้าภาพรวม)

### ข้อสรุป
- **Top 10 อธิบาย return รายวันของ S&P ได้ ~80%** — CV 0.78–0.83 ทุกโมเดล; test AutoGluon/XGBoost 0.81–0.83; สูงสุดในปีวิกฤต (2018/2020/2022 = 0.89–0.93)
- **Sensitivity ขึ้นกับ regime** — ดัชนีขยับ ~0.7–0.9% ต่อ 1% ของ top 10 ในปีผันผวน แต่ ~0.5% ในปี 2023–24 (ต่ำสุดในรอบ 10 ปี): mega-cap ขยับด้วยเรื่องตัวเองมากขึ้น ตลาดตามน้อยลง
- **Importance กระจุกที่อันดับ 7–10** — ผลของ collinearity ในกลุ่มเทค + อันดับ 7–10 เป็น proxy ของหุ้นอีก 490 ตัว ไม่ใช่น้ำหนัก market cap
- **Linear vs non-linear** — ใน CV ใกล้กัน; บน test tree/ensemble นำ ~0.1 เพราะจับ non-linearity ได้ (SHAP dependence รูป S: การขยับเกิน ±4% ของหุ้นตัวเดียวส่งผลต่อดัชนีน้อยกว่าสัดส่วนเชิงเส้น) และความต่างนี้มีนัยสำคัญ (AutoGluon/XGBoost/SVR/Ridge: ΔR² 95% CI ไม่คร่อม 0, Diebold–Mariano p < 0.001; LSTM ไม่ต่าง) แต่ใน fold COVID Linear กลับดีที่สุดเพราะ extrapolate ได้
- **LSTM ≈ Linear** — CV เท่ากัน (0.778 vs 0.782) test ต่างกันอย่างไม่มีนัยสำคัญ: return รายวันแทบไม่มี autocorrelation ประวัติย้อนหลังไม่ช่วย

### ข้อจำกัด
- ไม่มี ticker → ตีความได้ระดับอันดับ; การจับคู่เป็นการอนุมาน (ตรวจสอบแล้วด้วยคู่ราคาใกล้กัน, R² ก่อน/หลัง และ sensitivity ของพารามิเตอร์)
- ไม่มี market cap → แยกน้ำหนักจริงออกจาก co-movement ไม่ได้ (coefficient/importance เป็น sensitivity เชิงสถิติ)
- test เป็นช่วงเดียว (2024–25) — อ่านคู่กับ CV และช่วงความเชื่อมั่นจาก bootstrap
''')
    st.markdown('---')
    st.subheader('คำถามที่อาจถูกถาม — และคำตอบจากผลในงานนี้')
    qa = [
        ('Top 10 อยู่ในดัชนีอยู่แล้ว R² สูงจึงเป็นเรื่องธรรมดาไม่ใช่หรือ?',
         'ใช่ส่วนหนึ่ง — top 10 มีน้ำหนักในดัชนี ~35–40% (2024–25) ส่วนนั้นเป็นกลไกที่หลีกเลี่ยงไม่ได้ แต่ R² ~0.8 สูงกว่าน้ำหนักมาก ส่วนต่างคือ co-movement ของหุ้นอีก 490 ตัว '
         'งานนี้จึงวัด "อิทธิพลเชิงสถิติ" และระบุชัดว่าการแยก mechanical weight ต้องมี market cap'),
        ('ทำไมไม่ทำนายอนาคต (t+1)?',
         'โจทย์คือ "อิทธิพล" ไม่ใช่การพยากรณ์ และหน้า *ข้อมูล* แสดงว่าถ้าใช้ return ของ top 10 เมื่อวานทำนายวันนี้ R² ≈ 0 ตามทฤษฎีตลาดมีประสิทธิภาพ — ตัวเลข same-day จึงไม่ใช่ผลของ look-ahead'),
        ('ทำไมไม่ใช้ราคา (level) ที่ normalize แล้ว?',
         'level ไม่ stationary (ADF p ≈ 0.98) → regression เป็น spurious (placebo "ลำดับวัน" อย่างเดียวได้ R² 0.88, Durbin-Watson 0.14) และ XGBoost บน level ได้ test R² = −4.1 เพราะ extrapolate ไม่ได้; return stationary (ADF p ≈ 0), DW ≈ 2'),
        ('การ un-swap เชื่อถือได้แค่ไหน ในเมื่อไม่มี ticker?',
         'ตรวจ 4 ทาง: คู่หุ้นราคาใกล้กัน (GOOGL/GOOG) ได้ correlation 0.976; R² test จาก −1.9 → 0.71; การขยับใหญ่ที่เก็บไว้ตรงกับเหตุการณ์จริงที่ทราบวันที่ (META −19% 26/7/2018, NVDA +24% 25/5/2023, META +20% 2/2/2024, NVDA −17% 27/1/2025); '
         'และตาราง sensitivity: ขยับพารามิเตอร์ทุกตัวแล้วผลระดับรวมแทบไม่เปลี่ยน — กฎ boundary jump >20% ตั้งจากการตรวจ event ปลอม (TSLA/AVGO สลับกันที่อันดับ 10 ปี 2024) กับราคาจริง ไม่ใช่การจูนกับ R² test และแสดงผลเมื่อปิดกฎไว้ด้วย'),
        ('โมเดลไหนต้อง normalize ใช้สูตรอะไร และทำไม?',
         'ทุกโมเดลถูกปฏิบัติเหมือนกัน: ชนิด scaler เป็น hyperparameter ที่ CV เลือกจาก none / z-score (x−μ)/σ / min-max (x−min)/(max−min) โดย scaler fit บน train ของแต่ละ fold เท่านั้น และใช้กติกาเดียวกัน (CV สูงสุด; ถ้า none ตามหลังไม่เกิน 0.001 ให้ none) — '
         'ผล (ตารางในหน้า Hyperparameter search): Linear ทั้ง 3 แบบเท่ากันถึงทศนิยมที่ 15 (OLS scale-equivariant), XGBoost ทุก config ใน search ต่างกัน < 1e-5 (tree แบ่ง node ด้วยลำดับของค่า), AutoGluon none ดีที่สุดอยู่แล้ว (ต่างกัน 0.002; มี preprocessing ของตัวเองต่อโมเดล) → ทั้งสามเลือก none; '
         'Ridge (+0.017), SVR (+0.024) และ LSTM (+0.046) ดีขึ้นชัดเจนเมื่อ scale และ CV เลือก min-max ทั้งสามตัว: return มีหางหนา (−21% ถึง +24%) z-score ทำให้วันสุดขั้วมี |z| 5–10 (ระยะทางใน RBF ระเบิด / gate ของ LSTM อิ่มตัว) ส่วน min-max บีบทุกค่าลง [0,1]; '
         'Ridge penalty ขึ้นกับหน่วย ชนิด scaler จึงเปลี่ยนรูปแบบการ shrink — coefficient แปลงกลับเป็นหน่วยเดิมเพื่อตีความ; target ไม่ scale (หน่วย %)'),
        ('Hyperparameter ถูกเลือกโดยเห็น test ไหม?',
         'ไม่ — ทุก search ใช้ค่าเฉลี่ย R² ของ 4 val fold เท่านั้น แล้ว fit ใหม่บน train ทั้งหมด วัด test ครั้งเดียว; LSTM early stopping และ AutoGluon tuning ใช้ส่วนท้ายของ train (อดีต); scaler fit ต่อ fold'),
        ('ค่า val ราย fold optimistic ไหม?',
         'เล็กน้อยสำหรับโมเดลที่ search เพราะ config ถูกเลือกจาก fold เหล่านั้น — จึงใช้ test เป็นตัวเลขหลัก และให้ช่วงความเชื่อมั่น (block bootstrap) กับ Diebold–Mariano test ในหน้า *เปรียบเทียบโมเดล*'),
        ('AutoGluon/XGBoost ดีกว่า Linear จริงหรือแค่โชคของช่วง test?',
         'ดูตาราง significance ในหน้า *เปรียบเทียบโมเดล*: CI ของ ΔR² ที่ไม่คร่อม 0 และ DM p < 0.05 = ต่างจาก Linear อย่างมีนัยสำคัญบน test นี้ — อ่านค่าจากตารางโดยตรง'),
        ('ทำไม importance ไม่เรียงตาม market cap?',
         'อันดับ 1–5 correlate กันสูง (VIF 3–4, corr กับกลุ่ม 0.74–0.81) จึงแชร์เครดิต ส่วนอันดับ 7–10 เป็น proxy ของหุ้นอีก 490 ตัว — เช็คแล้วว่าไม่ได้เกิดจากการ impute (หน้า *Feature importance*)'),
        ('Linear ชนะทุกโมเดลใน fold COVID ได้อย่างไร?',
         'วันที่ตลาด ±10% เกินช่วงที่ train เคยเห็น tree/LSTM ให้ค่าทำนายแบน (extrapolate ไม่ได้: XGBoost 0.73, LSTM 0.69, AutoGluon 0.68) แต่ linear extrapolate ได้ตามโครงสร้างดัชนีที่เป็นผลรวมถ่วงน้ำหนัก (0.84) '
         'และ SVR ที่ CV เลือก γ เล็กจน kernel เกือบเชิงเส้นก็ตามมาติดๆ (0.82) เช่นเดียวกับ Ridge (0.80)'),
        ('LSTM ทำไมไม่ดีกว่า Linear ทั้งที่ซับซ้อนกว่า?',
         'return รายวันแทบไม่มี autocorrelation (corr ของ EW top 10 เมื่อวานกับดัชนีวันนี้ ≈ −0.13) ประวัติย้อนหลังจึงไม่เพิ่มข้อมูล แต่เพิ่มพารามิเตอร์และความแปรปรวน — CV เท่ากับ Linear (0.778 vs 0.782) '
         'และบน test สูงกว่า (0.769 vs 0.707) แต่ไม่มีนัยสำคัญ (block bootstrap: CI ของ ΔR² คร่อม 0, DM p = 0.16); early stopping หยุดเร็ว (~14 epoch)'),
        ('MAPE ทำไมคำนวณบนระดับดัชนี?',
         'สูตร MAPE หารด้วยค่าจริง ซึ่ง return รายวันใกล้ 0 บ่อยมาก (12% ของวัน |return| < 0.1%) ทำให้ได้ตัวเลขหลายร้อย % ที่ไม่สื่ออะไร จึงแปลง return ที่ทำนายเป็นระดับดัชนีก่อนตามธรรมเนียมงานทำนายราคา — MAPE ระดับดัชนี ≈ MAE ของ return'),
        ('Split 4 fold × 365 วันซื้อขาย ทำไมไม่แบ่งตามปี?',
         'เป็นการออกแบบตามแผนงาน (train 80% / test 20%, expanding window); 365 วันซื้อขาย ≈ 1 ปี 5 เดือน; ผลรายปีมีในหน้า *อิทธิพลตามเวลา* ให้ดูประกอบ'),
        ('ข้อมูล adj close มีผลไหม?',
         'adj close ปรับปันผลย้อนหลัง ทำให้ระดับไม่ตรงราคาซื้อขายจริง แต่ return รายวันต่างเฉพาะวัน ex-dividend (~0.5% ต่อไตรมาส) ผลน้อยมาก; target (S&P 500) เป็น close จริง ตรวจกับค่าที่ทราบแล้ว (เช่น 23/3/2020 = 2,237.40)'),
        ('ทำไม SVR ใช้ kernel เดียว (RBF)?',
         'RBF ถูกกำหนดไว้ล่วงหน้าเพราะเป็น kernel มาตรฐานที่ bounded (ค่าทำนายไม่โตไม่จำกัดเมื่อ input สุดขั้ว) — หน้า Hyperparameter search เทียบ linear / poly / sigmoid ด้วย protocol เดียวกัน (min-max ตามที่ CV เลือกให้ RBF): '
         'CV mean ต่างกันไม่เกิน 0.004 (แยกไม่ออกด้วย CV; linear และ RBF เท่ากันที่ 0.830 เพราะ γ = 0.0005 บน input [0,1] เล็กมาก RBF จึงเกือบเชิงเส้น) บน test อยู่ในช่วง 0.78–0.81 แคบกว่า CI ของ R² มาก และ stress test ไม่มี kernel ไหนระเบิด '
         'ดังนั้น (1) ยอมรับว่า RBF เป็นทางเลือกล่วงหน้า (2) ถ้า search kernel ด้วย CV ก็ได้ RBF/linear เท่ากัน (3) ผลและข้อสรุปหลักไม่ขึ้นกับ kernel'),
        ('ผลนี้ generalize ไปอนาคตได้ไหม?',
         'test เป็นช่วงเดียว (2024–25) และ sensitivity ขึ้นกับ regime จึงควรรายงานเป็น "ช่วง 2016–2025" และ re-fit เมื่อมีข้อมูลใหม่ ไม่ควรอ้างเป็นค่าคงที่'),
    ]
    for q, a in qa:
        with st.expander('Q: ' + q):
            st.write(a)


# ----------------------------------------------------------------------------- main
if not DATA_PATH.exists():
    st.error(f'ไม่พบ {DATA_PATH.name} — รัน sp500_top10_influence.ipynb ให้จบ (section 16 จะสร้างไฟล์นี้) แล้ว refresh')
    st.stop()
B = load_bundle(str(DATA_PATH), DATA_PATH.stat().st_mtime)

PAGES = {
    'ภาพรวม': page_overview,
    'เปรียบเทียบโมเดล': page_compare,
    'Feature importance': page_importance,
    'อิทธิพลตามเวลา': page_time,
    'ข้อมูลและ un-swap': page_data,
    'Hyperparameter search': page_search,
    'วิธีการและข้อสรุป': page_method,
}
with st.sidebar:
    st.header('Top-10 vs S&P 500')
    page = st.radio('หน้า', list(PAGES), label_visibility='collapsed')
    st.markdown('---')
    st.caption(f"โมเดล: {', '.join(B['meta']['models'])}")
    st.caption(f"รันเมื่อ {B['meta']['run_at']}")
PAGES[page](B)
