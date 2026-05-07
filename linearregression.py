import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  피처 설정 — 여기만 수정하면 됩니다
#
#  새 피처 추가 방법:
#   1. 아래 리스트에 딕셔너리 블록 하나 복붙
#   2. file / date_col / columns 세 줄만 내 파일에 맞게 수정
#   3. 바로 실행 → FEATURES 자동 갱신
#
# [필드 설명]
#   file      : CSV 경로
#   date_col  : 날짜 컬럼명
#   date_fmt  : 날짜 형식 (None이면 pandas 자동 파싱)
#               예) "%Y-%m-%d" / "%d-%b-%Y" / None
#   columns   : 가져올 피처 컬럼명 리스트
#   agg       : 일별→월별 집계 방식 "mean" / "sum" / "last"
#
# ※ 컬럼명 변경·쉼표 제거 등 특수 전처리가 필요한 파일은
#   섹션 2에서 _clean.csv 로 저장한 뒤 여기서 _clean 경로를 참조.
#   (현재 BDI, GSCPI 가 해당)
#
# ※ Linear Regression은 피처 간 단위 차이가 크면 계수가 왜곡됩니다.
#   새 피처 추가 시 StandardScaler가 자동 적용되므로 별도 처리 불필요.
# ═══════════════════════════════════════════════════════════════

FEATURE_CONFIG = [
    {
        "file":     "data/raw/oil_geopolitics_dataset_2010_2026.csv",
        "date_col": "date",
        "date_fmt": None,
        "columns":  ["wti_price", "brent_price", "dxy_index", "vix"],
        "agg":      "mean",
    },
    {
        "file":     "data/raw/gscpi_data_clean.csv",   
        "date_col": "Date",
        "date_fmt": "%d-%b-%Y",
        "columns":  ["GSCPI"],
        "agg":      "mean",
    },
    {
        "file":     "data/raw/Baltic_Dry_Index_clean.csv", 
        "date_col": "Date",
        "date_fmt": None,
        "columns":  ["BDI"],
        "agg":      "mean",
    },
    {
        "file":     "data/raw/data_gpr_export.csv",
        "date_col": "month",
        "date_fmt": None,
        "columns":  ["GPR", "GPRH"],
        "agg":      "mean",
    },

    # ── 새 피처 추가 예시 ─────────────────────────────────────────
    # 아래 블록을 복붙한 뒤 세 줄만 수정하세요
    #
    # {
    #     "file":     "data/raw/새파일.csv",   ← 파일 경로
    #     "date_col": "date",                  ← 날짜 컬럼명
    #     "date_fmt": None,                    ← 날짜 형식 (모르면 None)
    #     "columns":  ["새컬럼1", "새컬럼2"],  ← 가져올 컬럼명
    #     "agg":      "mean",                  ← 집계 방식
    # },
    # ─────────────────────────────────────────────────────────────
]

# ─────────────────────────────────────────────
# 1. 타겟 로드 (Air Freight Index)
# ─────────────────────────────────────────────
air_df = pd.read_csv("data/raw/Air_Freight_Index.csv")
air_df['observation_date'] = pd.to_datetime(air_df['observation_date'])
air_df['YearMonth'] = air_df['observation_date'].dt.to_period('M')
air_df = air_df.rename(columns={'PCU481112481112': 'Air_Freight_Index'})
air_monthly = air_df[['YearMonth', 'Air_Freight_Index']].copy()

# ─────────────────────────────────────────────
# 2. 특수 전처리 → _clean.csv 저장
# ─────────────────────────────────────────────

# BDI: 한글 컬럼명 변경 + 쉼표 제거
bdi_raw = pd.read_csv("data/raw/Baltic_Dry_Index.csv")
bdi_raw = bdi_raw.rename(columns={'날짜': 'Date', '종가': 'BDI'})
bdi_raw['BDI'] = bdi_raw['BDI'].astype(str).str.replace(',', '').astype(float)
bdi_raw[['Date', 'BDI']].dropna().to_csv("data/raw/Baltic_Dry_Index_clean.csv", index=False)

# GSCPI: 상단 4줄 메타데이터 제거 + 컬럼명 정리
gscpi_raw = pd.read_csv("data/raw/gscpi_data.csv", skiprows=4, usecols=[0, 1])
gscpi_raw.columns = ['Date', 'GSCPI']
gscpi_raw.dropna(subset=['Date', 'GSCPI']).to_csv("data/raw/gscpi_data_clean.csv", index=False)

# ─────────────────────────────────────────────
# 3. 자동 로딩 & 병합 루프
# ─────────────────────────────────────────────
df = air_monthly.copy()

for cfg in FEATURE_CONFIG:
    tmp = pd.read_csv(cfg["file"], usecols=[cfg["date_col"]] + cfg["columns"])
    tmp[cfg["date_col"]] = pd.to_datetime(
        tmp[cfg["date_col"]], format=cfg["date_fmt"], errors="coerce"
    )
    tmp = tmp.dropna(subset=[cfg["date_col"]])
    tmp["YearMonth"] = tmp[cfg["date_col"]].dt.to_period("M")

    monthly = (
        tmp.groupby("YearMonth")[cfg["columns"]]
        .agg(cfg["agg"])
        .reset_index()
    )
    df = pd.merge(df, monthly, on="YearMonth", how="inner")

df = df.sort_values('YearMonth').reset_index(drop=True)
df = df.dropna()

# FEATURE_CONFIG 에서 자동 수집
FEATURES = [col for cfg in FEATURE_CONFIG for col in cfg["columns"]]

# ─────────────────────────────────────────────
# 4. Feature / Target 분리 & Train/Test Split
# ─────────────────────────────────────────────
X = df[FEATURES]
y = df['Air_Freight_Index']
dates = df['YearMonth'].astype(str)

X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
    X, y, dates, test_size=0.3, random_state=42
)

# ── 스케일링 ─────────
# 피처 간 단위 차이가 크므로 (VIX: 10~80 vs BDI: 300~11000)
# StandardScaler로 평균 0, 표준편차 1로 정규화
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 5. 모델 학습
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"📊 [Linear Regression 평가 결과]")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.2f} 포인트")
print(f" RMSE :  {rmse:.2f}")

coef_df = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print("\n📌 회귀 계수 (스케일링 후):")
print(coef_df.to_string(index=False))

# ─────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────
COLOR = '#2196F3'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0F0F1A')

for ax in [ax1, ax2]:
    ax.set_facecolor('#1A1A2E')
    ax.tick_params(colors='#CCCCCC', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# ── 산점도 ────────────────────────────────────
ax1.scatter(y_test, y_pred, color=COLOR, alpha=0.65,
            edgecolors='white', linewidths=0.4, s=70)
lim_min = min(y_test.min(), y_pred.min()) * 0.95
lim_max = max(y_test.max(), y_pred.max()) * 1.05
ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'w--', lw=1.5, alpha=0.7)
ax1.set_xlabel('Actual Air Freight Index', color='#AAAACC', fontsize=10)
ax1.set_ylabel('Predicted Air Freight Index', color='#AAAACC', fontsize=10)
ax1.set_title(f'Actual vs Predicted (Scatter)\nR² = {r2:.3f}  |  MAE = {mae:.2f}',
              color='white', fontsize=12, fontweight='bold')
ax1.text(0.05, 0.93,
         'Points on the dashed line\nmean perfect prediction!',
         transform=ax1.transAxes, fontsize=9, color='white', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#2A2A4A', alpha=0.8))

# ── 시계열 비교 ───────────────────────────────
sort_idx   = np.argsort(d_test.values)
dates_sort = d_test.values[sort_idx]
actual_s   = y_test.values[sort_idx]
pred_s     = y_pred[sort_idx]

x_pos     = np.arange(len(dates_sort))
tick_step = max(1, len(x_pos) // 8)

ax2.plot(x_pos, actual_s, color='white', lw=1.8, label='Actual',    alpha=0.9)
ax2.plot(x_pos, pred_s,   color=COLOR,   lw=1.8, label='Predicted',
         linestyle='--', alpha=0.9)
ax2.fill_between(x_pos, actual_s, pred_s, color=COLOR, alpha=0.12)
ax2.set_xticks(x_pos[::tick_step])
ax2.set_xticklabels(dates_sort[::tick_step], rotation=35, ha='right',
                     fontsize=8, color='#AAAACC')
ax2.set_ylabel('Air Freight Index', color='#AAAACC', fontsize=10)
ax2.set_title('Actual vs Predicted (Time Series)',
              color='white', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.3, labelcolor='white',
           facecolor='#2A2A4A', edgecolor='#444466')

fig.suptitle('Linear Regression — Air Freight Index Prediction',
             color='white', fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('result/lr_results.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

# ── 계수 중요도 ───────────────────────────────
coef_sorted = pd.Series(np.abs(model.coef_), index=FEATURES).sort_values(ascending=True)

fig2, ax = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#0F0F1A')
ax.set_facecolor('#1A1A2E')
ax.barh(coef_sorted.index, coef_sorted.values, color=COLOR,
        alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_xlabel('|Coefficient| (StandardScaler 적용 후)', color='#AAAACC', fontsize=10)
ax.set_title('Linear Regression — Feature Coefficient',
             color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='#CCCCCC', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig('result/lr_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()

