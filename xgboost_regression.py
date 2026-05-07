import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =========================================================
# 0. 경로 설정
# =========================================================
# 이 .py 파일이 있는 위치 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dataset csv 파일은 data/raw/ 안에 넣기
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# 그래프 이미지는 .py 파일과 같은 위치에 저장
OUTPUT_DIR = BASE_DIR


# =========================================================
# 1. 피처 설정
# =========================================================
# Linear Regression 코드와 같은 피처를 사용해야 모델 간 공정 비교 가능
FEATURE_CONFIG = [
    {
        "file": "oil_geopolitics_dataset_2010_2026.csv",
        "date_col": "date",
        "date_fmt": None,
        "columns": ["wti_price", "brent_price", "dxy_index", "vix"],
        "agg": "mean",
    },
    {
        "file": "gscpi_data_clean.csv",
        "date_col": "Date",
        "date_fmt": "%d-%b-%Y",
        "columns": ["GSCPI"],
        "agg": "mean",
    },
    {
        "file": "Baltic_Dry_Index_clean.csv",
        "date_col": "Date",
        "date_fmt": None,
        "columns": ["BDI"],
        "agg": "mean",
    },
    {
        "file": "data_gpr_export.csv",
        "date_col": "month",
        "date_fmt": None,
        "columns": ["GPR", "GPRH"],
        "agg": "mean",
    },
]


# =========================================================
# 2. 데이터 로드 함수
# =========================================================
def load_monthly_feature(cfg):
    """
    각 csv 파일을 불러와 월별 데이터로 변환하는 함수
    """
    file_path = os.path.join(DATA_DIR, cfg["file"])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    use_cols = [cfg["date_col"]] + cfg["columns"]
    tmp = pd.read_csv(file_path, usecols=use_cols)

    tmp[cfg["date_col"]] = pd.to_datetime(
        tmp[cfg["date_col"]],
        format=cfg["date_fmt"],
        errors="coerce"
    )

    tmp = tmp.dropna(subset=[cfg["date_col"]])
    tmp["YearMonth"] = tmp[cfg["date_col"]].dt.to_period("M")

    monthly = (
        tmp.groupby("YearMonth")[cfg["columns"]]
        .agg(cfg["agg"])
        .reset_index()
    )

    return monthly


# =========================================================
# 3. 타겟 데이터 로드
# =========================================================
air_path = os.path.join(DATA_DIR, "Air_Freight_Index.csv")

if not os.path.exists(air_path):
    raise FileNotFoundError(f"Air_Freight_Index.csv 파일이 없습니다: {air_path}")

air_df = pd.read_csv(air_path)

air_df["observation_date"] = pd.to_datetime(air_df["observation_date"])
air_df["YearMonth"] = air_df["observation_date"].dt.to_period("M")

air_df = air_df.rename(columns={"PCU481112481112": "Air_Freight_Index"})
air_monthly = air_df[["YearMonth", "Air_Freight_Index"]].copy()


# =========================================================
# 4. 피처 데이터 병합
# =========================================================
df = air_monthly.copy()

for cfg in FEATURE_CONFIG:
    monthly_feature = load_monthly_feature(cfg)
    df = pd.merge(df, monthly_feature, on="YearMonth", how="inner")

df = df.sort_values("YearMonth").reset_index(drop=True)
df = df.dropna()

FEATURES = [col for cfg in FEATURE_CONFIG for col in cfg["columns"]]

print("=" * 70)
print("XGBoost Regression - Air Freight Index Prediction")
print("=" * 70)
print(f"데이터 기간: {df['YearMonth'].min()} ~ {df['YearMonth'].max()}")
print(f"데이터 개수: {len(df)}개월")
print(f"사용 피처 {len(FEATURES)}개: {FEATURES}")
print()


# =========================================================
# 5. Feature / Target 분리
# =========================================================
X = df[FEATURES]
y = df["Air_Freight_Index"]
dates = df["YearMonth"].astype(str)

# 기존 Linear Regression과 동일하게 7:3 split
X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
    X,
    y,
    dates,
    test_size=0.3,
    random_state=42
)

print(f"Train: {len(X_train)}개월")
print(f"Test : {len(X_test)}개월")
print()


# =========================================================
# 6. XGBoost 모델 학습
# =========================================================
# XGBoost는 트리 기반 모델이라 StandardScaler가 필수는 아님
# Linear Regression과 달리 변수 단위 차이에 덜 민감함
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# scenario_2026_jan = pd.DataFrame([{
#     "wti_price": 64.765,
#     "brent_price": 60.259,
#     "dxy_index": 98.2,
#     "vix": 16,
#     "GSCPI": 0.2,
#     "BDI": 1777.48,
#     "GPR": 130.8,
#     "GPRH": 110.0,
# }])

scenario_2026_jan = pd.DataFrame([{
    "wti_price":   69.4, # 2
    "brent_price": 64.52, # 2
    "dxy_index":   97.45, # 2
    "vix":         19.23, # 2
    "GSCPI":       0.54, # 2
    "BDI":         2040.9, # 2
    "GPR":         121.62, # 2
    "GPRH":        108.26, # 2
}])

actual_2026_jan = 175.010

pred_2026_jan = model.predict(scenario_2026_jan)[0]

error = actual_2026_jan - pred_2026_jan
abs_error = abs(error)
error_rate = abs_error / actual_2026_jan * 100
accuracy = 100 - error_rate

print("[2026년 1월 실제값 검증 - GDP X]")
print(f"실제값      : {actual_2026_jan:.3f}")
print(f"예측값      : {pred_2026_jan:.3f}")
print(f"오차        : {error:.3f}")
print(f"절대오차    : {abs_error:.3f}")
print(f"오차율      : {error_rate:.2f}%")
print(f"정확도      : {accuracy:.2f}%")

# =========================================================
# 7. 성능 평가
# =========================================================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("[XGBoost Regression 평가 결과]")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
print()

# 성능 결과 저장
metrics_df = pd.DataFrame({
    "Model": ["XGBoost Regression"],
    "R2_Score": [r2],
    "MAE": [mae],
    "RMSE": [rmse],
    "Train_Size": [len(X_train)],
    "Test_Size": [len(X_test)]
})

metrics_path = os.path.join(OUTPUT_DIR, "xgb_metrics.csv")
metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")


# =========================================================
# 8. 예측 결과 저장
# =========================================================
result_df = pd.DataFrame({
    "YearMonth": d_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Error": y_test.values - y_pred,
    "Abs_Error": np.abs(y_test.values - y_pred)
})

result_df = result_df.sort_values("YearMonth").reset_index(drop=True)

pred_path = os.path.join(OUTPUT_DIR, "xgb_predictions.csv")
result_df.to_csv(pred_path, index=False, encoding="utf-8-sig")


# =========================================================
# 9. 피처 중요도 정리
# =========================================================
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("[XGBoost 피처 중요도]")
print(importance_df.to_string(index=False))
print()

importance_path = os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv")
importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")


# =========================================================
# 10. 시각화 1 - 실제값 vs 예측값
# =========================================================
COLOR = "#4CAF50"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#0F0F1A")

for ax in [ax1, ax2]:
    ax.set_facecolor("#1A1A2E")
    ax.tick_params(colors="#CCCCCC", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

# 산점도
ax1.scatter(
    y_test,
    y_pred,
    color=COLOR,
    alpha=0.7,
    edgecolors="white",
    linewidths=0.4,
    s=70
)

lim_min = min(y_test.min(), y_pred.min()) * 0.95
lim_max = max(y_test.max(), y_pred.max()) * 1.05

ax1.plot(
    [lim_min, lim_max],
    [lim_min, lim_max],
    "w--",
    lw=1.5,
    alpha=0.7
)

ax1.set_xlabel("Actual Air Freight Index", color="#AAAACC", fontsize=10)
ax1.set_ylabel("Predicted Air Freight Index", color="#AAAACC", fontsize=10)
ax1.set_title(
    f"Actual vs Predicted Scatter\nR² = {r2:.3f} | MAE = {mae:.2f} | RMSE = {rmse:.2f}",
    color="white",
    fontsize=12,
    fontweight="bold"
)

ax1.text(
    0.05,
    0.93,
    "Points close to dashed line\nmean better prediction",
    transform=ax1.transAxes,
    fontsize=9,
    color="white",
    va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#2A2A4A", alpha=0.8)
)

# 시계열 비교
sort_idx = np.argsort(d_test.values)
dates_sort = d_test.values[sort_idx]
actual_s = y_test.values[sort_idx]
pred_s = y_pred[sort_idx]

x_pos = np.arange(len(dates_sort))
tick_step = max(1, len(x_pos) // 8)

ax2.plot(
    x_pos,
    actual_s,
    color="white",
    lw=1.8,
    label="Actual",
    alpha=0.9
)

ax2.plot(
    x_pos,
    pred_s,
    color=COLOR,
    lw=1.8,
    label="Predicted",
    linestyle="--",
    alpha=0.9
)

ax2.fill_between(
    x_pos,
    actual_s,
    pred_s,
    color=COLOR,
    alpha=0.12
)

ax2.set_xticks(x_pos[::tick_step])
ax2.set_xticklabels(
    dates_sort[::tick_step],
    rotation=35,
    ha="right",
    fontsize=8,
    color="#AAAACC"
)

ax2.set_ylabel("Air Freight Index", color="#AAAACC", fontsize=10)
ax2.set_title(
    "Actual vs Predicted Time Series",
    color="white",
    fontsize=12,
    fontweight="bold"
)

ax2.legend(
    fontsize=9,
    framealpha=0.3,
    labelcolor="white",
    facecolor="#2A2A4A",
    edgecolor="#444466"
)

fig.suptitle(
    "XGBoost Regression - Air Freight Index Prediction",
    color="white",
    fontsize=14,
    fontweight="bold",
    y=1.01
)

plt.tight_layout()

result_img_path = os.path.join(OUTPUT_DIR, "xgb_results.png")
plt.savefig(
    result_img_path,
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
plt.close()


# =========================================================
# 11. 시각화 2 - 피처 중요도
# =========================================================
importance_sorted = importance_df.sort_values("Importance", ascending=True)

fig2, ax = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor("#0F0F1A")
ax.set_facecolor("#1A1A2E")

ax.barh(
    importance_sorted["Feature"],
    importance_sorted["Importance"],
    color=COLOR,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4
)

ax.set_xlabel("Feature Importance", color="#AAAACC", fontsize=10)
ax.set_title(
    "XGBoost Regression - Feature Importance",
    color="white",
    fontsize=12,
    fontweight="bold"
)

ax.tick_params(colors="#CCCCCC", labelsize=9)

for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

plt.tight_layout()

importance_img_path = os.path.join(OUTPUT_DIR, "xgb_feature_importance.png")
plt.savefig(
    importance_img_path,
    dpi=150,
    bbox_inches="tight",
    facecolor=fig2.get_facecolor()
)
plt.close()


# =========================================================
# 12. 저장 완료 출력
# =========================================================
print("저장 완료")
print(f"- {result_img_path}")
print(f"- {importance_img_path}")
print(f"- {metrics_path}")
print(f"- {pred_path}")
print(f"- {importance_path}")