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

# 2026년 2월 실제 입력 피처값
# 원본 파일에 주석으로 들어있던 값이며, GDP는 연간 데이터라 1월과 동일하게 설정
SCENARIO_2026_FEB = {
    "wti_price": 69.4,
    "brent_price": 64.52,
    "dxy_index": 97.45,
    "vix": 19.23,
    "GSCPI": 0.54,
    "BDI": 2040.9,
    "GPR": 121.62,
    "GPRH": 108.26,
    "GDP": 1.23e14,
}

SCENARIO_2026_MAR = {
    "wti_price": 99.405,      # 실제로는 Brent 2026년 3월 월평균
    "brent_price": 91.380,    # 실제로는 WTI 2026년 3월 월평균
    "dxy_index": 99.01,
    "vix": 25.597,
    "GSCPI": 0.677,
    "BDI": 2064.0,
    "GPR": 297.27,
    "GPRH": 251.38,
    "GDP": 1.23e14,
}

# 2026년 1월 / 2월 실제 Air Freight Index
# Air_Freight_Index.csv 안에 해당 월 값이 있으면 그 값을 우선 사용합니다.
DEFAULT_ACTUAL_2026_JAN = 177.227
DEFAULT_ACTUAL_2026_FEB = 175.010  # 2월 실제값을 알고 있다면 여기에 숫자 입력

# =========================================================
# 그래프 기준선 통일 설정
# =========================================================
# GDP O 모델은 GDP 데이터가 2024-12까지 병합되므로
# GDP X 그래프도 과거 차트를 2024-12까지만 보여주고,
# 기준선을 2025-01-01로 고정해 두 그래프의 양식을 통일합니다.
COMMON_HISTORY_END = pd.Period("2024-12", freq="M")
COMMON_DIVIDER_DATE = pd.Timestamp("2025-01-01")
FUTURE_DATE_2026_JAN = pd.Timestamp("2026-01-01")
FUTURE_DATE_2026_FEB = pd.Timestamp("2026-02-01")


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


def get_actual_for_month(period_str, default=None):
    """
    Air_Freight_Index.csv에 해당 월 값이 있으면 그 값을 사용하고,
    없으면 default 값을 반환합니다 (None 가능).
    """
    air_monthly = load_air_freight()
    target_month = pd.Period(period_str, freq="M")

    actual_row = air_monthly[air_monthly["YearMonth"] == target_month]

    if len(actual_row) > 0:
        return float(actual_row["Air_Freight_Index"].iloc[0])

    return default


def predict_one_month(model, features, scenario_dict, actual_value, month_label):
    """
    단일 월에 대해 예측을 수행하고 분석 결과를 dict로 반환합니다.
    actual_value가 None이면 오차 관련 항목은 None으로 채워집니다.
    """
    scenario_input = pd.DataFrame([
        {feature: scenario_dict[feature] for feature in features}
    ])

    pred = float(model.predict(scenario_input)[0])

    if actual_value is not None:
        error = actual_value - pred
        abs_error = abs(error)
        error_rate = abs_error / actual_value * 100
        accuracy = 100 - error_rate
    else:
        error = None
        abs_error = None
        error_rate = None
        accuracy = None

    return {
        "Month": month_label,
        "Actual": actual_value,
        "Predicted": pred,
        "Error": error,
        "Abs_Error": abs_error,
        "Error_Rate_%": error_rate,
        "Accuracy_%": accuracy,
    }


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

    # 2026년 1월 / 2월 입력값 예측
    actual_jan = get_actual_for_month("2026-01", default=DEFAULT_ACTUAL_2026_JAN)
    actual_feb = get_actual_for_month("2026-02", default=DEFAULT_ACTUAL_2026_FEB)

    pred_jan = predict_one_month(model, features, SCENARIO_2026_JAN, actual_jan, "2026-01")
    pred_feb = predict_one_month(model, features, SCENARIO_2026_FEB, actual_feb, "2026-02")

    result = {
        "Model": model_name,
        "GDP": "O" if with_gdp else "X",
        "Feature_Count": len(features),
        "Data_Period": f"{df['YearMonth'].min()} ~ {df['YearMonth'].max()}",
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "predictions": {
            "2026-01": pred_jan,
            "2026-02": pred_feb,
        },
        # 1월 결과 (하위 호환)
        "Actual_2026_Jan": pred_jan["Actual"],
        "Predicted_2026_Jan": pred_jan["Predicted"],
        "Error_Jan": pred_jan["Error"],
        "Abs_Error_Jan": pred_jan["Abs_Error"],
        "Error_Rate_Jan_%": pred_jan["Error_Rate_%"],
        "Accuracy_Jan_%": pred_jan["Accuracy_%"],
        # 2월 결과
        "Actual_2026_Feb": pred_feb["Actual"],
        "Predicted_2026_Feb": pred_feb["Predicted"],
        "Error_Feb": pred_feb["Error"],
        "Abs_Error_Feb": pred_feb["Abs_Error"],
        "Error_Rate_Feb_%": pred_feb["Error_Rate_%"],
        "Accuracy_Feb_%": pred_feb["Accuracy_%"],
    }

    return model, df, features, result


def plot_history_and_2026_prediction(df, result, output_name):
    """
    한 장의 그래프 안에서
    왼쪽: 과거 실제값 + 모델 예측선
    오른쪽: 2026년 1월/2월 실제값 + 1월/2월 예측값 차이
    를 보여줍니다.
    """
    model_name = result["Model"]

    plot_df = df.copy()

    # -----------------------------------------------------
    # 기준선 통일 핵심 수정
    # -----------------------------------------------------
    plot_df = plot_df[plot_df["YearMonth"] <= COMMON_HISTORY_END].copy()
    plot_df = plot_df.reset_index(drop=True)

    if plot_df.empty:
        raise ValueError(
            f"그래프에 표시할 데이터가 없습니다. COMMON_HISTORY_END={COMMON_HISTORY_END} 값을 확인하세요."
        )

    plot_df["Date"] = plot_df["YearMonth"].dt.to_timestamp()

    jan_date = FUTURE_DATE_2026_JAN
    # feb_date = FUTURE_DATE_2026_FEB
    last_date = plot_df["Date"].iloc[-1]
    gap = jan_date - last_date
    feb_date = jan_date + gap

    pred_jan = result["predictions"]["2026-01"]
    pred_feb = result["predictions"]["2026-02"]

    actual_jan = pred_jan["Actual"]
    predicted_jan = pred_jan["Predicted"]
    actual_feb = pred_feb["Actual"]
    predicted_feb = pred_feb["Predicted"]

    if result["GDP"] == "O":
        model_color = "#FF6B35"
    else:
        model_color = "#4CAF50"

    fig, ax = plt.subplots(figsize=(17, 7))
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
    divider_date = COMMON_DIVIDER_DATE
    ax.axvline(
        divider_date,
        color="#DDDDDD",
        linewidth=2.0,
        alpha=0.85
    )

    # -----------------------------------------------------
    # 이후 예측: 실제값 라인 (white)
    # 마지막 과거값 → 1월 실제값 → (가능하면) 2월 실제값 까지 연결
    # -----------------------------------------------------
    actual_line_x = [last_date, jan_date]
    actual_line_y = [plot_df["Air_Freight_Index"].iloc[-1], actual_jan]

    if actual_feb is not None:
        actual_line_x.append(feb_date)
        actual_line_y.append(actual_feb)

    ax.plot(
        actual_line_x,
        actual_line_y,
        color="white",
        linewidth=2.2,
        alpha=0.9
    )

    # -----------------------------------------------------
    # 이후 예측: 모델 예측 라인 (colored dashed)
    # 마지막 모델 예측값 → 1월 예측값 → 2월 예측값 까지 연결
    # -----------------------------------------------------
    ax.plot(
        [last_date, jan_date, feb_date],
        [plot_df["Model_Predicted"].iloc[-1], predicted_jan, predicted_feb],
        color=model_color,
        linewidth=2.5,
        linestyle="--",
        alpha=0.95
    )

    # -----------------------------------------------------
    # 1월 포인트
    # -----------------------------------------------------
    ax.scatter(
        [jan_date],
        [actual_jan],
        color="white",
        s=130,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="2026년 실제값 (1월/2월)"
    )

    ax.scatter(
        [jan_date],
        [predicted_jan],
        color=model_color,
        s=140,
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
        label="2026년 예측값 (1월/2월)"
    )

    # 1월 실제 vs 예측 오차
    ax.vlines(
        jan_date,
        ymin=min(actual_jan, predicted_jan),
        ymax=max(actual_jan, predicted_jan),
        colors="#FFD166",
        linestyles=":",
        linewidth=3,
        label="2026년 예측 오차"
    )

    # -----------------------------------------------------
    # 2월 포인트
    # -----------------------------------------------------
    if actual_feb is not None:
        ax.scatter(
            [feb_date],
            [actual_feb],
            color="white",
            s=130,
            edgecolors="black",
            linewidths=1.0,
            zorder=5
        )

    ax.scatter(
        [feb_date],
        [predicted_feb],
        color=model_color,
        s=140,
        edgecolors="white",
        linewidths=1.0,
        zorder=6
    )

    # 2월 실제 vs 예측 오차
    if actual_feb is not None:
        ax.vlines(
            feb_date,
            ymin=min(actual_feb, predicted_feb),
            ymax=max(actual_feb, predicted_feb),
            colors="#FFD166",
            linestyles=":",
            linewidth=3
        )

    # -----------------------------------------------------
    # 라벨 (이전 / 2026년 예측)
    # -----------------------------------------------------
    y_min, y_max = ax.get_ylim()
    ax.text(
        plot_df["Date"].iloc[int(len(plot_df) * 0.45)],
        y_min + (y_max - y_min) * 0.08,
        "이전 차트",
        color="#DDDDDD",
        fontsize=14,
        ha="center"
    )

    midpoint = jan_date + (feb_date - jan_date) / 2
    ax.text(
        midpoint,
        y_min + (y_max - y_min) * 0.08,
        "2026년 1월~2월 예측",
        color="#DDDDDD",
        fontsize=14,
        ha="center"
    )

    # -----------------------------------------------------
    # 1월 분석 결과 어노테이션
    # -----------------------------------------------------
    if actual_jan is not None and pred_jan["Abs_Error"] is not None:
        jan_annotation = (
            f"[2026년 1월]\n"
            f"실제값: {actual_jan:.3f}\n"
            f"예측값: {predicted_jan:.3f}\n"
            f"절대오차: {pred_jan['Abs_Error']:.3f}\n"
            f"오차율: {pred_jan['Error_Rate_%']:.2f}%\n"
            f"정확도: {pred_jan['Accuracy_%']:.2f}%"
        )
    else:
        jan_annotation = (
            f"[2026년 1월]\n"
            f"예측값: {predicted_jan:.3f}\n"
            f"(실제값 없음)"
        )

    ax.annotate(
        jan_annotation,
        xy=(jan_date, predicted_jan),
        xytext=(jan_date + pd.DateOffset(months=5), predicted_jan + (y_max - y_min) * 0.25),
        color="white",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=model_color, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A2A4A", edgecolor=model_color, alpha=0.9)
    )

    # -----------------------------------------------------
    # 2월 분석 결과 어노테이션
    # -----------------------------------------------------
    if actual_feb is not None and pred_feb["Abs_Error"] is not None:
        feb_annotation = (
            f"[2026년 2월]\n"
            f"실제값: {actual_feb:.3f}\n"
            f"예측값: {predicted_feb:.3f}\n"
            f"절대오차: {pred_feb['Abs_Error']:.3f}\n"
            f"오차율: {pred_feb['Error_Rate_%']:.2f}%\n"
            f"정확도: {pred_feb['Accuracy_%']:.2f}%"
        )
    else:
        feb_annotation = (
            f"[2026년 2월]\n"
            f"예측값: {predicted_feb:.3f}\n"
            f"(실제값 미입력)"
        )

    ax.annotate(
        feb_annotation,
        xy=(feb_date, predicted_feb),
        xytext=(feb_date + pd.DateOffset(months=4), predicted_feb - (y_max - y_min) * 0.20),
        color="white",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#FFD166", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A2A4A", edgecolor="#FFD166", alpha=0.9)
    )

    # -----------------------------------------------------
    # 모델 성능 지표 (좌상단)
    # -----------------------------------------------------
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
        f"{model_name} - 과거 실제값/모델 예측선 + 2026년 1월·2월 실제값 검증",
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
    # 2월 이후로 어노테이션 공간 확보
    ax.set_xlim(plot_df["Date"].min(), feb_date + pd.DateOffset(months=10))

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"그래프 저장 완료: {save_path}")


def plot_compare_bar(results_df):
    """
    GDP X / GDP O의 2026년 1월·2월 실제값과 예측값 차이를 막대그래프로 비교합니다.
    실제값이 없는 월은 예측값만 표시합니다.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    labels = []
    actuals = []
    preds = []
    abs_errors = []
    error_rates = []

    for _, row in results_df.iterrows():
        # 1월
        labels.append(f"{row['Model']}\n(1월)")
        actuals.append(row["Actual_2026_Jan"])
        preds.append(row["Predicted_2026_Jan"])
        abs_errors.append(row["Abs_Error_Jan"])
        error_rates.append(row["Error_Rate_Jan_%"])
        # 2월
        labels.append(f"{row['Model']}\n(2월)")
        actuals.append(row["Actual_2026_Feb"])
        preds.append(row["Predicted_2026_Feb"])
        abs_errors.append(row["Abs_Error_Feb"])
        error_rates.append(row["Error_Rate_Feb_%"])

    x = np.arange(len(labels))
    width = 0.35

    actual_mask = [a is not None and not pd.isna(a) for a in actuals]

    actual_label_used = False
    pred_label_used = False
    for i in range(len(labels)):
        if actual_mask[i]:
            ax.bar(
                x[i] - width / 2,
                actuals[i],
                width,
                color="white",
                edgecolor="black",
                label="실제값" if not actual_label_used else None
            )
            actual_label_used = True
        ax.bar(
            x[i] + width / 2,
            preds[i],
            width,
            color="#FF6B35",
            edgecolor="white",
            alpha=0.9,
            label="예측값" if not pred_label_used else None
        )
        pred_label_used = True

    # 값 라벨 표시
    for i in range(len(labels)):
        if actual_mask[i]:
            ax.text(
                x[i] - width / 2,
                actuals[i] + 1,
                f"{actuals[i]:.2f}",
                ha="center",
                color="white",
                fontsize=10
            )
        ax.text(
            x[i] + width / 2,
            preds[i] + 1,
            f"{preds[i]:.2f}",
            ha="center",
            color="white",
            fontsize=10
        )

        if actual_mask[i] and abs_errors[i] is not None and not pd.isna(abs_errors[i]):
            base_y = min(actuals[i], preds[i]) - 10
            ax.text(
                x[i],
                base_y,
                f"절대오차 {abs_errors[i]:.2f}\n오차율 {error_rates[i]:.2f}%",
                ha="center",
                color="#FFD166",
                fontsize=9
            )
        else:
            ax.text(
                x[i],
                preds[i] - 10,
                "(실제값 미입력)",
                ha="center",
                color="#AAAACC",
                fontsize=9
            )

    ax.set_title(
        "2026년 1월·2월 Air Freight Index 실제값 vs 예측값 비교",
        color="white",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Air Freight Index", color="#AAAACC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#CCCCCC", fontsize=10)
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

    save_path = os.path.join(OUTPUT_DIR, "xgb_2026_jan_feb_actual_vs_pred_compare.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"비교 그래프 저장 완료: {save_path}")


if __name__ == "__main__":
    model_x, df_x, features_x, result_x = train_xgb_and_predict_2026(with_gdp=False)
    model_o, df_o, features_o, result_o = train_xgb_and_predict_2026(with_gdp=True)

    plot_history_and_2026_prediction(
        df=df_x,
        result=result_x,
        output_name="xgb_2026_jan_feb_chart_gdp_x_fixed.png"
    )

    plot_history_and_2026_prediction(
        df=df_o,
        result=result_o,
        output_name="xgb_2026_jan_feb_chart_gdp_o_fixed.png"
    )

    summary_columns = [
        "Model", "GDP", "Feature_Count", "Data_Period", "R2", "MAE", "RMSE",
        "Actual_2026_Jan", "Predicted_2026_Jan", "Error_Jan", "Abs_Error_Jan",
        "Error_Rate_Jan_%", "Accuracy_Jan_%",
        "Actual_2026_Feb", "Predicted_2026_Feb", "Error_Feb", "Abs_Error_Feb",
        "Error_Rate_Feb_%", "Accuracy_Feb_%",
    ]
    results_df = pd.DataFrame([
        {col: result_x.get(col) for col in summary_columns},
        {col: result_o.get(col) for col in summary_columns},
    ])
    summary_path = os.path.join(OUTPUT_DIR, "xgb_2026_jan_feb_validation_summary.csv")
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_compare_bar(results_df)

    print("\n" + "=" * 80)
    print("2026년 1월·2월 실제값 검증 결과")
    print("=" * 80)

    # 1월 결과 출력
    print("\n[2026년 1월 예측 결과]")
    print(
        results_df[
            [
                "Model",
                "Feature_Count",
                "R2",
                "MAE",
                "RMSE",
                "Actual_2026_Jan",
                "Predicted_2026_Jan",
                "Error_Jan",
                "Abs_Error_Jan",
                "Error_Rate_Jan_%",
                "Accuracy_Jan_%",
            ]
        ].round(4).to_string(index=False)
    )

    # 2월 결과 출력
    print("\n[2026년 2월 예측 결과]")
    print(
        results_df[
            [
                "Model",
                "Feature_Count",
                "Actual_2026_Feb",
                "Predicted_2026_Feb",
                "Error_Feb",
                "Abs_Error_Feb",
                "Error_Rate_Feb_%",
                "Accuracy_Feb_%",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n저장 파일")
    print("- xgb_2026_jan_feb_chart_gdp_x_fixed.png")
    print("- xgb_2026_jan_feb_chart_gdp_o_fixed.png")
    print("- xgb_2026_jan_feb_actual_vs_pred_compare.png")
    print("- xgb_2026_jan_feb_validation_summary.csv")
