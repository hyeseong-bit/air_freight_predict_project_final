import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Random Forest Regression — GDP O / GDP X 결과 분리 이미지 생성
#
# 핵심:
#   기존 randomforest.py와 같은 파라미터/피처/Train-Test Split을 사용한다.
#   단, 결과 이미지만 GDP X / GDP O로 각각 따로 저장한다.
#
# 실행 위치:
#   프로젝트 루트(gangin)에서 실행
#   python models/randomforest_gdp_o_x_scatter_timeseries_fixed_params.py
#
# 저장 결과:
#   results/randomforest_scatter_timeseries_fixed/rf_gdp_x_scatter_timeseries_fixed.png
#   results/randomforest_scatter_timeseries_fixed/rf_gdp_o_scatter_timeseries_fixed.png
#   results/randomforest_scatter_timeseries_fixed/rf_gdp_o_x_metrics_fixed.csv
#   results/randomforest_scatter_timeseries_fixed/rf_gdp_o_x_prediction_compare_fixed.csv
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR
RAW_DIR = PROJECT_DIR / "data" / "raw"
RESULT_DIR = PROJECT_DIR / "results" / "randomforest_scatter_timeseries_fixed"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 기존 randomforest.py와 동일한 설정
RANDOM_STATE = 42
TEST_SIZE = 0.3

GDP_COUNTRY_CODE = "WLD"
GDP_FEATURE_NAME = "World_GDP_log"
FILL_GDP_MISSING_YEAR_WITH_LATEST = True

FEATURE_CONFIG = [
    {
        "file":     RAW_DIR / "oil_geopolitics_dataset_2010_2026.csv",
        "date_col": "date",
        "date_fmt": None,
        "columns":  ["wti_price", "brent_price", "dxy_index", "vix"],
        "agg":      "mean",
    },
    {
        "file":     RAW_DIR / "gscpi_data_clean.csv",
        "date_col": "Date",
        "date_fmt": "%d-%b-%Y",
        "columns":  ["GSCPI"],
        "agg":      "mean",
    },
    {
        "file":     RAW_DIR / "Baltic_Dry_Index_clean.csv",
        "date_col": "Date",
        "date_fmt": None,
        "columns":  ["BDI"],
        "agg":      "mean",
    },
    {
        "file":     RAW_DIR / "data_gpr_export.csv",
        "date_col": "month",
        "date_fmt": None,
        "columns":  ["GPR", "GPRH"],
        "agg":      "mean",
    },
]

# 그래프 스타일
COLOR_BG = "#0F0F1A"
COLOR_AX = "#1A1A2E"
COLOR_GRID = "#6B6B85"
COLOR_TEXT = "#D8D8E8"
COLOR_WHITE = "#F5F5F5"
COLOR_GREEN = "#59B95B"
COLOR_ORANGE = "#FF6B35"
COLOR_DIAGONAL = "#B8B8C8"


# ═══════════════════════════════════════════════════════════════
# 0. 보조 함수
# ═══════════════════════════════════════════════════════════════

def set_korean_font():
    """Windows/맥/리눅스에서 한글이 깨지지 않도록 사용 가능한 폰트를 자동 선택한다."""
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def safe_read_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    return pd.read_csv(path, **kwargs)


def ensure_clean_files():
    """BDI, GSCPI clean 파일이 없을 경우 자동 생성한다."""
    bdi_clean = RAW_DIR / "Baltic_Dry_Index_clean.csv"
    bdi_raw_path = RAW_DIR / "Baltic_Dry_Index.csv"

    if not bdi_clean.exists() and bdi_raw_path.exists():
        bdi_raw = pd.read_csv(bdi_raw_path)
        bdi_raw = bdi_raw.rename(columns={"날짜": "Date", "종가": "BDI"})
        if "Date" not in bdi_raw.columns or "BDI" not in bdi_raw.columns:
            raise ValueError("BDI 파일에서 Date/BDI 컬럼을 찾지 못했습니다.")
        bdi_raw["BDI"] = (
            bdi_raw["BDI"].astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )
        bdi_raw[["Date", "BDI"]].dropna().to_csv(bdi_clean, index=False)

    gscpi_clean = RAW_DIR / "gscpi_data_clean.csv"
    gscpi_raw_path = RAW_DIR / "gscpi_data.csv"

    if not gscpi_clean.exists() and gscpi_raw_path.exists():
        try:
            gscpi_raw = pd.read_csv(gscpi_raw_path)
            if "Date" in gscpi_raw.columns and "GSCPI" in gscpi_raw.columns:
                gscpi_raw[["Date", "GSCPI"]].dropna().to_csv(gscpi_clean, index=False)
            else:
                raise ValueError
        except Exception:
            gscpi_raw = pd.read_csv(gscpi_raw_path, skiprows=4, usecols=[0, 1])
            gscpi_raw.columns = ["Date", "GSCPI"]
            gscpi_raw.dropna(subset=["Date", "GSCPI"]).to_csv(gscpi_clean, index=False)


def load_air_freight_target():
    air_path = RAW_DIR / "Air_Freight_Index.csv"
    air_df = safe_read_csv(air_path)

    air_df["observation_date"] = pd.to_datetime(air_df["observation_date"], errors="coerce")
    air_df = air_df.dropna(subset=["observation_date"])
    air_df["YearMonth"] = air_df["observation_date"].dt.to_period("M")

    if "PCU481112481112" in air_df.columns:
        air_df = air_df.rename(columns={"PCU481112481112": "Air_Freight_Index"})

    if "Air_Freight_Index" not in air_df.columns:
        raise ValueError("Air_Freight_Index 또는 PCU481112481112 컬럼을 찾지 못했습니다.")

    air_df["Air_Freight_Index"] = pd.to_numeric(air_df["Air_Freight_Index"], errors="coerce")
    return air_df[["YearMonth", "Air_Freight_Index"]].dropna().copy()


def add_config_features(df):
    for cfg in FEATURE_CONFIG:
        usecols = [cfg["date_col"]] + cfg["columns"]
        tmp = safe_read_csv(cfg["file"], usecols=usecols)

        tmp[cfg["date_col"]] = pd.to_datetime(
            tmp[cfg["date_col"]],
            format=cfg["date_fmt"],
            errors="coerce"
        )
        tmp = tmp.dropna(subset=[cfg["date_col"]])
        tmp["YearMonth"] = tmp[cfg["date_col"]].dt.to_period("M")

        for col in cfg["columns"]:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        monthly = (
            tmp.groupby("YearMonth")[cfg["columns"]]
            .agg(cfg["agg"])
            .reset_index()
        )
        df = pd.merge(df, monthly, on="YearMonth", how="inner")

    return df


def load_gdp_monthly(year_months):
    gdp_path = RAW_DIR / "gdp.csv"
    if not gdp_path.exists():
        fallback_gdp_path = PROJECT_DIR / "gdp.csv"
        if fallback_gdp_path.exists():
            gdp_path = fallback_gdp_path
        else:
            raise FileNotFoundError(f"gdp.csv 파일이 없습니다: {gdp_path}")

    gdp_raw = pd.read_csv(gdp_path, skiprows=4)

    if "Country Code" not in gdp_raw.columns:
        raise ValueError("gdp.csv에서 Country Code 컬럼을 찾지 못했습니다.")

    row = gdp_raw[gdp_raw["Country Code"] == GDP_COUNTRY_CODE]
    if row.empty:
        available = gdp_raw["Country Code"].dropna().unique()[:20]
        raise ValueError(f"GDP_COUNTRY_CODE={GDP_COUNTRY_CODE} 행을 찾지 못했습니다. 예시 코드: {available}")

    year_cols = [c for c in gdp_raw.columns if str(c).isdigit()]
    gdp_long = row.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name="GDP"
    )

    gdp_long["Year"] = gdp_long["Year"].astype(int)
    gdp_long["GDP"] = pd.to_numeric(gdp_long["GDP"], errors="coerce")
    gdp_long = gdp_long.sort_values("Year")

    target_years = pd.Series(year_months).astype(str).str[:4].astype(int)
    year_frame = pd.DataFrame({
        "Year": range(target_years.min(), target_years.max() + 1)
    })

    gdp_yearly = pd.merge(year_frame, gdp_long[["Year", "GDP"]], on="Year", how="left")

    if FILL_GDP_MISSING_YEAR_WITH_LATEST:
        gdp_yearly["GDP"] = gdp_yearly["GDP"].ffill()

    gdp_yearly[GDP_FEATURE_NAME] = np.log(gdp_yearly["GDP"])

    monthly = pd.DataFrame({"YearMonth": year_months})
    monthly["Year"] = monthly["YearMonth"].astype(str).str[:4].astype(int)
    monthly = pd.merge(monthly, gdp_yearly[["Year", GDP_FEATURE_NAME]], on="Year", how="left")
    monthly = monthly[["YearMonth", GDP_FEATURE_NAME]]

    return monthly


def build_dataset():
    ensure_clean_files()

    df = load_air_freight_target()
    df = add_config_features(df)

    df = df.sort_values("YearMonth").reset_index(drop=True)
    df = df.dropna()

    gdp_monthly = load_gdp_monthly(df["YearMonth"])
    df = pd.merge(df, gdp_monthly, on="YearMonth", how="left")
    df = df.dropna()

    return df


# ═══════════════════════════════════════════════════════════════
# 1. 모델 학습
# ═══════════════════════════════════════════════════════════════

def train_random_forest(df, features, train_idx, test_idx, label):
    X = df[features]
    y = df["Air_Freight_Index"]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # 기존 randomforest.py와 같은 Random Forest 파라미터
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": label,
        "Feature_Count": len(features),
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    pred_df = pd.DataFrame({
        "YearMonth": df["YearMonth"].iloc[test_idx].astype(str).values,
        "Actual": y_test.values,
        "Predicted": y_pred,
    }).sort_values("YearMonth").reset_index(drop=True)

    return model, metrics, pred_df


# ═══════════════════════════════════════════════════════════════
# 2. 그래프 생성
# ═══════════════════════════════════════════════════════════════

def setup_dark_axis(ax):
    ax.set_facecolor(COLOR_AX)
    ax.grid(True, color=COLOR_GRID, alpha=0.28, linewidth=1)
    ax.tick_params(colors="#BBBBBB", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.xaxis.label.set_color("#BBBBBB")
    ax.yaxis.label.set_color("#BBBBBB")
    ax.title.set_color("white")


def plot_one_model(pred_df, metrics, label, color, out_filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    fig.patch.set_facecolor(COLOR_BG)

    ax1, ax2 = axes
    setup_dark_axis(ax1)
    setup_dark_axis(ax2)

    # ── 왼쪽: Actual vs Predicted Scatter ─────────────────────
    ax1.scatter(
        pred_df["Actual"],
        pred_df["Predicted"],
        s=58,
        color=color,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
    )

    lim_min = min(pred_df["Actual"].min(), pred_df["Predicted"].min()) * 0.95
    lim_max = max(pred_df["Actual"].max(), pred_df["Predicted"].max()) * 1.05
    ax1.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        color=COLOR_DIAGONAL,
        linewidth=1.5,
        alpha=0.85,
    )
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)

    ax1.set_title(
        f"Actual vs Predicted Scatter\nR² = {metrics['R2']:.4f} | MAE = {metrics['MAE']:.2f} | RMSE = {metrics['RMSE']:.2f}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Actual Air Freight Index", fontsize=10)
    ax1.set_ylabel("Predicted Air Freight Index", fontsize=10)

    ax1.text(
        0.05,
        0.86,
        "Points close to dashed line\nmean better prediction",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        color=COLOR_TEXT,
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#252543",
            edgecolor="#444466",
            alpha=0.85,
        ),
    )

    # ── 오른쪽: 테스트셋 실제값/예측값 시계열 ─────────────────
    x_pos = np.arange(len(pred_df))
    tick_step = max(1, len(x_pos) // 8)

    ax2.plot(
        x_pos,
        pred_df["Actual"],
        color=COLOR_WHITE,
        linewidth=2.0,
        label="Actual",
    )
    ax2.plot(
        x_pos,
        pred_df["Predicted"],
        color=color,
        linewidth=2.0,
        linestyle="--",
        label="Predicted",
    )

    ax2.set_title("Actual vs Predicted Time Series", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Air Freight Index", fontsize=10)
    ax2.set_xticks(x_pos[::tick_step])
    ax2.set_xticklabels(
        pred_df["YearMonth"].iloc[::tick_step],
        rotation=35,
        ha="right",
        fontsize=8,
        color="#AAAACC",
    )

    leg = ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor="#20203B",
        edgecolor="#333355",
        fontsize=9,
    )
    for text in leg.get_texts():
        text.set_color(COLOR_TEXT)

    fig.suptitle(
        f"Random Forest Regression - Air Freight Index Prediction ({label})",
        color="white",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = RESULT_DIR / out_filename
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    return out_path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    set_korean_font()

    df = build_dataset()

    base_features = [col for cfg in FEATURE_CONFIG for col in cfg["columns"]]
    gdp_features = base_features + [GDP_FEATURE_NAME]

    # 기존 randomforest.py와 동일하게, 먼저 전체 행 index를 한 번만 분리한다.
    # 이렇게 해야 GDP X / GDP O가 완전히 같은 Train/Test 행을 사용한다.
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"✅ 데이터셋: {len(df)}개월 ({df['YearMonth'].min()} ~ {df['YearMonth'].max()})")
    print(f"✅ 기존 randomforest.py와 같은 설정 사용")
    print(f"   - TEST_SIZE = {TEST_SIZE}")
    print(f"   - n_estimators = 500")
    print(f"   - max_depth = None")
    print(f"   - min_samples_leaf = 2")
    print(f"   - 시간 피처 month/year/t 사용 안 함")
    print(f"Train: {len(train_idx)}개월 / Test: {len(test_idx)}개월\n")

    model_no_gdp, metrics_no_gdp, pred_no_gdp = train_random_forest(
        df, base_features, train_idx, test_idx, "GDP X"
    )

    model_with_gdp, metrics_with_gdp, pred_with_gdp = train_random_forest(
        df, gdp_features, train_idx, test_idx, "GDP O"
    )

    # CSV 저장
    metric_df = pd.DataFrame([
        {**metrics_no_gdp, "Model": "Random Forest + GDP X"},
        {**metrics_with_gdp, "Model": "Random Forest + GDP O"},
    ])
    metric_df.to_csv(RESULT_DIR / "rf_gdp_o_x_metrics_fixed.csv", index=False, encoding="utf-8-sig")

    pred_compare = pd.merge(
        pred_no_gdp.rename(columns={"Predicted": "Pred_GDP_X"}),
        pred_with_gdp[["YearMonth", "Predicted"]].rename(columns={"Predicted": "Pred_GDP_O"}),
        on="YearMonth",
        how="inner",
    ).sort_values("YearMonth").reset_index(drop=True)
    pred_compare.to_csv(RESULT_DIR / "rf_gdp_o_x_prediction_compare_fixed.csv", index=False, encoding="utf-8-sig")

    # 이미지 저장
    path_gdp_x = plot_one_model(
        pred_df=pred_no_gdp,
        metrics=metrics_no_gdp,
        label="GDP X",
        color=COLOR_GREEN,
        out_filename="rf_gdp_x_scatter_timeseries_fixed.png",
    )

    path_gdp_o = plot_one_model(
        pred_df=pred_with_gdp,
        metrics=metrics_with_gdp,
        label="GDP O",
        color=COLOR_ORANGE,
        out_filename="rf_gdp_o_scatter_timeseries_fixed.png",
    )

    print("📊 [Random Forest 평가 결과]")
    print(metric_df[["Model", "Feature_Count", "R2", "MAE", "RMSE"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n✅ 저장 완료")
    print(f"- {path_gdp_x}")
    print(f"- {path_gdp_o}")
    print(f"- {RESULT_DIR / 'rf_gdp_o_x_metrics_fixed.csv'}")
    print(f"- {RESULT_DIR / 'rf_gdp_o_x_prediction_compare_fixed.csv'}")


if __name__ == "__main__":
    main()
