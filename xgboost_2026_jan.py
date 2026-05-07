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
# 0. 기본 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = BASE_DIR

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 2026년 1월 실제 입력 피처값
SCENARIO_2026_JAN = {
    "wti_price": 64.765,
    "brent_price": 60.259,
    "dxy_index": 98.2,
    "vix": 16,
    "GSCPI": 0.2,
    "BDI": 1777.48,
    "GPR": 130.8,
    "GPRH": 110.0,
    "GDP": 1.13e14,
}

# scenario_2026_FEB = pd.DataFrame([{
#     "wti_price":   69.4, # 2
#     "brent_price": 64.52, # 2
#     "dxy_index":   97.45, # 2
#     "vix":         19.23, # 2
#     "GSCPI":       0.54, # 2
#     "BDI":         2040.9, # 2
#     "GPR":         121.62, # 2
#     "GPRH":        108.26, # 2
# }])

# 2026년 1월 실제 Air Freight Index
# Air_Freight_Index.csv 안에 2026-01 값이 있으면 그 값을 우선 사용합니다.
DEFAULT_ACTUAL_2026_JAN = 177.227

# =========================================================
# 그래프 기준선 통일 설정
# =========================================================
# GDP O 모델은 GDP 데이터가 2024-12까지 병합되므로
# GDP X 그래프도 과거 차트를 2024-12까지만 보여주고,
# 기준선을 2025-01-01로 고정해 두 그래프의 양식을 통일합니다.
COMMON_HISTORY_END = pd.Period("2024-12", freq="M")
COMMON_DIVIDER_DATE = pd.Timestamp("2025-01-01")
FUTURE_DATE_2026_JAN = pd.Timestamp("2026-01-01")


def find_file(file_name):
    """
    data/raw 폴더에서 먼저 찾고, 없으면 현재 파이썬 파일이 있는 폴더에서 다시 찾습니다.
    """
    candidates = [
        os.path.join(DATA_DIR, file_name),
        os.path.join(BASE_DIR, file_name),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: {file_name}\n"
        f"확인 위치:\n- {candidates[0]}\n- {candidates[1]}"
    )


def prepare_gdp_clean():
    """
    GDP O 모델에서 사용할 gdp_clean.csv를 준비합니다.
    1순위: gdp_clean.csv가 있으면 그대로 사용
    2순위: gdp.csv 원본이 있으면 World GDP만 뽑아서 월별 데이터로 변환
    """
    gdp_clean_candidates = [
        os.path.join(DATA_DIR, "gdp_clean.csv"),
        os.path.join(BASE_DIR, "gdp_clean.csv"),
    ]

    for path in gdp_clean_candidates:
        if os.path.exists(path):
            check = pd.read_csv(path, nrows=5)
            if {"Date", "GDP"}.issubset(check.columns):
                return path

    gdp_raw_candidates = [
        os.path.join(DATA_DIR, "gdp.csv"),
        os.path.join(BASE_DIR, "gdp.csv"),
    ]

    gdp_raw_path = None
    for path in gdp_raw_candidates:
        if os.path.exists(path):
            gdp_raw_path = path
            break

    if gdp_raw_path is None:
        raise FileNotFoundError(
            "GDP 파일이 없습니다. data/raw 폴더에 gdp_clean.csv 또는 gdp.csv를 넣어주세요."
        )

    gdp_raw = pd.read_csv(gdp_raw_path, skiprows=4)

    if "Country Code" not in gdp_raw.columns:
        raise ValueError("gdp.csv에서 'Country Code' 컬럼을 찾을 수 없습니다.")

    gdp_world = gdp_raw[gdp_raw["Country Code"] == "WLD"].copy()

    if gdp_world.empty:
        raise ValueError("gdp.csv에서 Country Code가 'WLD'인 전세계 GDP 행을 찾지 못했습니다.")

    year_cols = [col for col in gdp_raw.columns if str(col).isdigit()]

    gdp_long = (
        gdp_world
        .melt(value_vars=year_cols, var_name="Year", value_name="GDP")
        .dropna(subset=["GDP"])
    )

    gdp_long["Year"] = gdp_long["Year"].astype(int)

    month_df = pd.DataFrame({"Month": range(1, 13)})
    gdp_monthly = gdp_long.merge(month_df, how="cross")

    gdp_monthly["Date"] = pd.to_datetime(
        gdp_monthly["Year"].astype(str)
        + "-"
        + gdp_monthly["Month"].astype(str).str.zfill(2)
        + "-01"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, "gdp_clean.csv")
    gdp_monthly[["Date", "GDP"]].to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[GDP 전처리 완료] {save_path}")
    return save_path


def get_feature_config(with_gdp=False):
    """
    with_gdp=False : 기존 XGBoost, GDP X, 8개 피처
    with_gdp=True  : GDP 추가 XGBoost, GDP O, 9개 피처
    """
    config = [
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

    if with_gdp:
        prepare_gdp_clean()
        config.append(
            {
                "file": "gdp_clean.csv",
                "date_col": "Date",
                "date_fmt": None,
                "columns": ["GDP"],
                "agg": "mean",
            }
        )

    return config


def load_monthly_feature(cfg):
    file_path = find_file(cfg["file"])
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


def load_air_freight():
    air_path = find_file("Air_Freight_Index.csv")
    air_df = pd.read_csv(air_path)

    air_df["observation_date"] = pd.to_datetime(air_df["observation_date"])
    air_df["YearMonth"] = air_df["observation_date"].dt.to_period("M")

    air_df = air_df.rename(columns={"PCU481112481112": "Air_Freight_Index"})
    air_monthly = air_df[["YearMonth", "Air_Freight_Index"]].copy()

    return air_monthly


def build_dataset(with_gdp=False):
    df = load_air_freight()
    feature_config = get_feature_config(with_gdp=with_gdp)

    for cfg in feature_config:
        monthly_feature = load_monthly_feature(cfg)
        df = pd.merge(df, monthly_feature, on="YearMonth", how="inner")

    df = df.sort_values("YearMonth").reset_index(drop=True)
    df = df.dropna()

    features = [col for cfg in feature_config for col in cfg["columns"]]

    return df, features


def get_actual_2026_jan():
    """
    Air_Freight_Index.csv에 2026-01 값이 있으면 그 값을 사용하고,
    없으면 DEFAULT_ACTUAL_2026_JAN 값을 사용합니다.
    """
    air_monthly = load_air_freight()
    target_month = pd.Period("2026-01", freq="M")

    actual_row = air_monthly[air_monthly["YearMonth"] == target_month]

    if len(actual_row) > 0:
        return float(actual_row["Air_Freight_Index"].iloc[0])

    return DEFAULT_ACTUAL_2026_JAN


def train_xgb_and_predict_2026(with_gdp=False):
    model_name = "XGBoost + GDP O" if with_gdp else "XGBoost + GDP X"

    df, features = build_dataset(with_gdp=with_gdp)

    X = df[features]
    y = df["Air_Freight_Index"]
    dates = df["YearMonth"].astype(str)

    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X,
        y,
        dates,
        test_size=0.3,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # 전체 과거 구간에 대한 모델 예측선
    df["Model_Predicted"] = model.predict(X)

    # 2026년 1월 입력값 예측
    scenario_input = pd.DataFrame([
        {feature: SCENARIO_2026_JAN[feature] for feature in features}
    ])

    actual_2026_jan = get_actual_2026_jan()
    pred_2026_jan = float(model.predict(scenario_input)[0])

    error = actual_2026_jan - pred_2026_jan
    abs_error = abs(error)
    error_rate = abs_error / actual_2026_jan * 100
    accuracy = 100 - error_rate

    result = {
        "Model": model_name,
        "GDP": "O" if with_gdp else "X",
        "Feature_Count": len(features),
        "Data_Period": f"{df['YearMonth'].min()} ~ {df['YearMonth'].max()}",
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Actual_2026_Jan": actual_2026_jan,
        "Predicted_2026_Jan": pred_2026_jan,
        "Error": error,
        "Abs_Error": abs_error,
        "Error_Rate_%": error_rate,
        "Accuracy_%": accuracy,
    }

    return model, df, features, result


def plot_history_and_2026_prediction(df, result, output_name):
    """
    한 장의 그래프 안에서
    왼쪽: 과거 실제값 + 모델 예측선
    오른쪽: 2026년 1월 실제값 + 2026년 1월 예측값 차이
    를 보여줍니다.
    """
    model_name = result["Model"]

    plot_df = df.copy()

    # -----------------------------------------------------
    # 기준선 통일 핵심 수정
    # -----------------------------------------------------
    # GDP X 모델은 더 긴 기간까지 데이터가 있을 수 있지만,
    # GDP O 모델과 같은 기준선으로 비교하기 위해
    # 두 그래프 모두 2024-12까지만 과거 차트로 표시합니다.
    plot_df = plot_df[plot_df["YearMonth"] <= COMMON_HISTORY_END].copy()
    plot_df = plot_df.reset_index(drop=True)

    if plot_df.empty:
        raise ValueError(
            f"그래프에 표시할 데이터가 없습니다. COMMON_HISTORY_END={COMMON_HISTORY_END} 값을 확인하세요."
        )

    plot_df["Date"] = plot_df["YearMonth"].dt.to_timestamp()

    future_date = FUTURE_DATE_2026_JAN
    last_date = plot_df["Date"].iloc[-1]

    actual_2026 = result["Actual_2026_Jan"]
    pred_2026 = result["Predicted_2026_Jan"]

    if result["GDP"] == "O":
        model_color = "#FF6B35"
    else:
        model_color = "#4CAF50"

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    # 이전 차트: 과거 실제값
    ax.plot(
        plot_df["Date"],
        plot_df["Air_Freight_Index"],
        color="white",
        linewidth=2.2,
        label="과거 실제값"
    )

    # 이전 차트: 생성된 모델의 예측 그래프
    ax.plot(
        plot_df["Date"],
        plot_df["Model_Predicted"],
        color=model_color,
        linewidth=2.0,
        linestyle="--",
        label="과거 모델 예측값"
    )

    # 이전 / 이후 구분선
    # GDP O 그래프처럼 기준선을 2025-01-01로 고정합니다.
    divider_date = COMMON_DIVIDER_DATE
    ax.axvline(
        divider_date,
        color="#DDDDDD",
        linewidth=2.0,
        alpha=0.85
    )

    # 이후 예측 차트: 마지막 과거값에서 2026년 1월 실제값까지 연결
    ax.plot(
        [last_date, future_date],
        [plot_df["Air_Freight_Index"].iloc[-1], actual_2026],
        color="white",
        linewidth=2.2,
        alpha=0.9
    )

    # 이후 예측 차트: 마지막 모델 예측값에서 2026년 1월 예측값까지 연결
    ax.plot(
        [last_date, future_date],
        [plot_df["Model_Predicted"].iloc[-1], pred_2026],
        color=model_color,
        linewidth=2.5,
        linestyle="--",
        alpha=0.95
    )

    # 2026년 1월 실제값 / 예측값 포인트
    ax.scatter(
        [future_date],
        [actual_2026],
        color="white",
        s=130,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="2026년 1월 실제값"
    )

    ax.scatter(
        [future_date],
        [pred_2026],
        color=model_color,
        s=140,
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
        label="2026년 1월 예측값"
    )

    # 실제값과 예측값 차이 표시
    ax.vlines(
        future_date,
        ymin=min(actual_2026, pred_2026),
        ymax=max(actual_2026, pred_2026),
        colors="#FFD166",
        linestyles=":",
        linewidth=3,
        label="2026년 1월 예측 오차"
    )

    # 라벨
    y_min, y_max = ax.get_ylim()
    ax.text(
        plot_df["Date"].iloc[int(len(plot_df) * 0.45)],
        y_min + (y_max - y_min) * 0.08,
        "이전 차트",
        color="#DDDDDD",
        fontsize=14,
        ha="center"
    )

    ax.text(
        future_date,
        y_min + (y_max - y_min) * 0.08,
        "2026년 1월 예측",
        color="#DDDDDD",
        fontsize=14,
        ha="center"
    )

    annotation = (
        f"실제값: {actual_2026:.3f}\n"
        f"예측값: {pred_2026:.3f}\n"
        f"절대오차: {result['Abs_Error']:.3f}\n"
        f"오차율: {result['Error_Rate_%']:.2f}%"
    )

    ax.annotate(
        annotation,
        xy=(future_date, pred_2026),
        xytext=(future_date + pd.DateOffset(months=4), pred_2026),
        color="white",
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=model_color, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A2A4A", edgecolor=model_color, alpha=0.9)
    )

    metric_text = (
        f"R² = {result['R2']:.4f}\n"
        f"MAE = {result['MAE']:.2f}\n"
        f"RMSE = {result['RMSE']:.2f}"
    )

    ax.text(
        0.02,
        0.96,
        metric_text,
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A2A4A", edgecolor="#555577", alpha=0.9)
    )

    ax.set_title(
        f"{model_name} - 과거 실제값/모델 예측선 + 2026년 1월 실제값 검증",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("날짜", color="#AAAACC", fontsize=11)
    ax.set_ylabel("Air Freight Index", color="#AAAACC", fontsize=11)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.78),
        fontsize=10,
        framealpha=0.25,
        labelcolor="white",
        facecolor="#2A2A4A",
        edgecolor="#444466"
    )

    ax.tick_params(colors="#CCCCCC", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.grid(True, alpha=0.18)
    ax.set_xlim(plot_df["Date"].min(), future_date + pd.DateOffset(months=9))

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"그래프 저장 완료: {save_path}")


def plot_compare_bar(results_df):
    """
    GDP X / GDP O의 2026년 1월 실제값과 예측값 차이를 막대그래프로 비교합니다.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    labels = results_df["Model"].tolist()
    actuals = results_df["Actual_2026_Jan"].values
    preds = results_df["Predicted_2026_Jan"].values

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, actuals, width, label="실제값", color="white", edgecolor="black")
    ax.bar(x + width / 2, preds, width, label="예측값", color="#FF6B35", edgecolor="white", alpha=0.9)

    for i in range(len(labels)):
        ax.text(x[i] - width / 2, actuals[i] + 1, f"{actuals[i]:.2f}", ha="center", color="white", fontsize=10)
        ax.text(x[i] + width / 2, preds[i] + 1, f"{preds[i]:.2f}", ha="center", color="white", fontsize=10)
        ax.text(
            x[i],
            min(actuals[i], preds[i]) - 8,
            f"절대오차 {results_df['Abs_Error'].iloc[i]:.2f}\n오차율 {results_df['Error_Rate_%'].iloc[i]:.2f}%",
            ha="center",
            color="#FFD166",
            fontsize=10
        )

    ax.set_title("2026년 1월 Air Freight Index 실제값 vs 예측값 비교", color="white", fontsize=14, fontweight="bold")
    ax.set_ylabel("Air Freight Index", color="#AAAACC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#CCCCCC")
    ax.tick_params(colors="#CCCCCC")

    ax.legend(
        fontsize=10,
        framealpha=0.3,
        labelcolor="white",
        facecolor="#2A2A4A",
        edgecolor="#444466"
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.grid(axis="y", alpha=0.18)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "xgb_2026_jan_actual_vs_pred_compare.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"비교 그래프 저장 완료: {save_path}")


if __name__ == "__main__":
    model_x, df_x, features_x, result_x = train_xgb_and_predict_2026(with_gdp=False)
    model_o, df_o, features_o, result_o = train_xgb_and_predict_2026(with_gdp=True)

    plot_history_and_2026_prediction(
        df=df_x,
        result=result_x,
        output_name="xgb_2026_jan_chart_gdp_x_fixed.png"
    )

    plot_history_and_2026_prediction(
        df=df_o,
        result=result_o,
        output_name="xgb_2026_jan_chart_gdp_o_fixed.png"
    )

    results_df = pd.DataFrame([result_x, result_o])
    summary_path = os.path.join(OUTPUT_DIR, "xgb_2026_jan_validation_summary.csv")
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_compare_bar(results_df)

    print("\n" + "=" * 80)
    print("2026년 1월 실제값 검증 결과")
    print("=" * 80)

    print(
        results_df[
            [
                "Model",
                "Feature_Count",
                "Data_Period",
                "R2",
                "MAE",
                "RMSE",
                "Actual_2026_Jan",
                "Predicted_2026_Jan",
                "Error",
                "Abs_Error",
                "Error_Rate_%",
                "Accuracy_%"
            ]
        ].round(4).to_string(index=False)
    )

    print("\n저장 파일")
    print("- xgb_2026_jan_chart_gdp_x_fixed.png")
    print("- xgb_2026_jan_chart_gdp_o_fixed.png")
    print("- xgb_2026_jan_actual_vs_pred_compare.png")
    print("- xgb_2026_jan_validation_summary.csv")
