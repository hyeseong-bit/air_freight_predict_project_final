import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 0. 실행 시작 메시지
# =========================================================
print("EDA 시작")

# =========================================================
# 1. 기본 경로 설정
# =========================================================
# 현재 파이썬 파일이 있는 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV 파일들이 들어있는 data 폴더
DATA_DIR = os.path.join(BASE_DIR, "data/raw")

# =========================================================
# 2. 그래프 전체 스타일 설정
# =========================================================
# 다크 배경 스타일 사용
plt.style.use("dark_background")

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"

# 마이너스 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# 전체 배경색
plt.rcParams["figure.facecolor"] = "#0F0F1A"

# 그래프 내부 배경색
plt.rcParams["axes.facecolor"] = "#1A1A2E"

# 축 색상
plt.rcParams["axes.edgecolor"] = "#555577"

# 축 라벨 색상
plt.rcParams["axes.labelcolor"] = "#DDDDDD"

# x축, y축 눈금 색상
plt.rcParams["xtick.color"] = "#CCCCCC"
plt.rcParams["ytick.color"] = "#CCCCCC"

# 격자 색상
plt.rcParams["grid.color"] = "#444466"

# =========================================================
# 3. 색상 변수
# =========================================================
# 타겟 강조 색상
TARGET_COLOR = "#FFFFFF"

# 주요 강조 색상
MAIN_COLOR = "#FF6B35"

# 보조 피처 색상
FEATURE_COLOR = "#4CAF50"

# 제목 강조 색상
TITLE_COLOR = "#FFD166"

# 축/설명 색상
TEXT_COLOR = "#DDDDDD"

# 그래프 배경색
FIG_BG = "#0F0F1A"
AX_BG = "#1A1A2E"


# =========================================================
# 4. 파일 찾기 함수
# =========================================================
def find_file(file_name):
    """
    data 폴더 안에서 CSV 파일을 찾는 함수
    예: data/Air_Freight_Index.csv
    """
    path = os.path.join(DATA_DIR, file_name)

    if os.path.exists(path):
        return path

    raise FileNotFoundError(f"{file_name} 파일이 없습니다. 위치 확인: {path}")


# =========================================================
# 5. 월 단위 데이터 변환 함수
# =========================================================
def load_monthly(file_name, date_col, cols):
    """
    서로 다른 CSV 파일을 월 단위로 맞추기 위한 함수

    file_name : 불러올 CSV 파일 이름
    date_col  : 날짜 컬럼 이름
    cols      : 사용할 피처 컬럼 리스트

    처리 과정:
    1. CSV 불러오기
    2. 날짜 컬럼을 datetime으로 변환
    3. 월 단위 YearMonth 생성
    4. 같은 월 데이터는 평균값으로 집계
    """
    temp = pd.read_csv(find_file(file_name), usecols=[date_col] + cols)

    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")

    temp = temp.dropna(subset=[date_col])

    temp["YearMonth"] = temp[date_col].dt.to_period("M")

    monthly = (
        temp.groupby("YearMonth")[cols]
        .mean()
        .reset_index()
    )

    return monthly


# =========================================================
# 6. x축 날짜 라벨 정리 함수
# =========================================================
def set_xticks(ax, df):
    """
    월 데이터가 너무 많으면 x축 글자가 겹침.
    그래서 전체 월 중 일부만 보이게 조절하는 함수.
    """
    tick_step = max(1, len(df) // 10)

    ax.set_xticks(range(0, len(df), tick_step))

    ax.set_xticklabels(
        df["YearMonth_str"].iloc[::tick_step],
        rotation=35,
        ha="right",
        fontsize=8
    )


# =========================================================
# 7. 그래프 공통 디자인 함수
# =========================================================
def style_axis(ax):
    """
    모든 그래프 축에 공통 디자인 적용
    """
    ax.set_facecolor(AX_BG)
    ax.grid(True, alpha=0.18)

    for spine in ax.spines.values():
        spine.set_color("#555577")

    ax.tick_params(colors="#CCCCCC")


# =========================================================
# 8. 타겟 데이터 불러오기
# =========================================================
# Air Freight Index는 예측 대상이 되는 y값
air = pd.read_csv(find_file("Air_Freight_Index.csv"))

# 날짜 컬럼 변환
air["observation_date"] = pd.to_datetime(
    air["observation_date"],
    errors="coerce"
)

# 월 단위 컬럼 생성
air["YearMonth"] = air["observation_date"].dt.to_period("M")

# 원래 컬럼명을 이해하기 쉬운 이름으로 변경
air = air.rename(columns={
    "PCU481112481112": "Air_Freight_Index"
})

# 필요한 컬럼만 선택
df = air[["YearMonth", "Air_Freight_Index"]].copy()


# =========================================================
# 9. 피처 파일 설정
# =========================================================
# 각 파일에서 어떤 날짜 컬럼과 어떤 피처를 사용할지 설정
feature_files = [
    (
        "oil_geopolitics_dataset_2010_2026.csv",
        "date",
        ["brent_price", "wti_price", "dxy_index", "vix"]
    ),
    (
        "gscpi_data_clean.csv",
        "Date",
        ["GSCPI"]
    ),
    (
        "Baltic_Dry_Index_clean.csv",
        "Date",
        ["BDI"]
    ),
    (
        "data_gpr_export.csv",
        "month",
        ["GPR", "GPRH"]
    ),
    (
        "gdp_clean.csv",
        "Date",
        ["GDP"]
    )
]


# =========================================================
# 10. 타겟 데이터와 피처 데이터 병합
# =========================================================
# 모든 데이터를 YearMonth 기준으로 inner join
for file_name, date_col, cols in feature_files:
    temp = load_monthly(file_name, date_col, cols)
    df = df.merge(temp, on="YearMonth", how="inner")


# =========================================================
# 11. 결측치 제거 및 최종 데이터 확인
# =========================================================
df = df.dropna().reset_index(drop=True)

if df.empty:
    raise ValueError("최종 데이터가 비었습니다. CSV 파일들의 날짜 범위가 서로 겹치는지 확인하세요.")

# 그래프 x축 라벨용 문자열 컬럼
df["YearMonth_str"] = df["YearMonth"].astype(str)

# 타겟 변수
target = "Air_Freight_Index"

# 입력 피처 9개
features = [
    "brent_price",
    "wti_price",
    "dxy_index",
    "vix",
    "GSCPI",
    "BDI",
    "GPR",
    "GPRH",
    "GDP"
]

print("데이터 shape:", df.shape)
print(df.head())


# =========================================================
# 12. 상관계수 Heatmap
# =========================================================
# 목적:
# Air Freight Index와 각 피처가 선형적으로 얼마나 같이 움직이는지 확인
corr = df[[target] + features].corr()

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(FIG_BG)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.6,
    linecolor="#222233",
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.8}
)

ax.set_title(
    "① Air Freight Index 기준 피처 상관계수",
    fontsize=17,
    fontweight="bold",
    color=TITLE_COLOR,
    pad=18
)

ax.tick_params(colors=TEXT_COLOR)
plt.tight_layout()
plt.show()


# =========================================================
# 13. 타겟 시계열 그래프
# =========================================================
# 목적:
# 예측 대상인 Air Freight Index 자체의 추세, 급등/급락 구간 확인
fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(FIG_BG)
style_axis(ax)

x = range(len(df))

# 실제 Air Freight Index
ax.plot(
    x,
    df[target],
    color=TARGET_COLOR,
    linewidth=2.3,
    label="Air Freight Index 실제 흐름"
)

# 6개월 이동평균
trend = df[target].rolling(6).mean()

ax.plot(
    x,
    trend,
    color=MAIN_COLOR,
    linewidth=2.6,
    linestyle="--",
    label="6개월 이동평균 추세"
)

set_xticks(ax, df)

ax.set_title(
    "② Air Freight Index 시계열 추세",
    fontsize=17,
    fontweight="bold",
    color=TITLE_COLOR,
    pad=18
)

ax.set_xlabel("월", color=TEXT_COLOR)
ax.set_ylabel("Air Freight Index", color=TEXT_COLOR)

ax.legend(
    facecolor="#2A2A4A",
    edgecolor="#666688",
    labelcolor=TEXT_COLOR,
    fontsize=10
)

plt.tight_layout()
plt.show()


# =========================================================
# 14. 타겟 vs 피처 산점도
# =========================================================
# 목적:
# 각 피처가 증가할 때 Air Freight Index가 어떻게 변하는지 확인
fig, axes = plt.subplots(3, 3, figsize=(17, 12))
fig.patch.set_facecolor(FIG_BG)
axes = axes.flatten()

for i, col in enumerate(features):
    ax = axes[i]
    style_axis(ax)

    ax.scatter(
        df[col],
        df[target],
        color=MAIN_COLOR,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
        s=45
    )

    ax.set_title(
        f"{col} → Air Freight Index",
        fontsize=12,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=10
    )

    ax.set_xlabel(col, color=TEXT_COLOR)
    ax.set_ylabel("Air Freight Index", color=TEXT_COLOR)

fig.suptitle(
    "③ 피처별 Air Freight Index와의 관계",
    fontsize=18,
    fontweight="bold",
    color=TITLE_COLOR,
    y=1.02
)

plt.tight_layout()
plt.show()


# =========================================================
# 15. 전체 변수 시계열 비교
# =========================================================
# 목적:
# 타겟과 9개 피처의 월별 흐름을 한 화면에서 비교
fig, axes = plt.subplots(5, 2, figsize=(18, 14))
fig.patch.set_facecolor(FIG_BG)
axes = axes.flatten()

all_cols = [target] + features

for i, col in enumerate(all_cols):
    ax = axes[i]
    style_axis(ax)

    # 타겟은 흰색, 피처는 초록색
    line_color = TARGET_COLOR if col == target else FEATURE_COLOR

    ax.plot(
        range(len(df)),
        df[col],
        color=line_color,
        linewidth=2.0,
        label=col
    )

    # 피처에는 6개월 이동평균선을 추가해서 흐름을 더 잘 보이게 함
    if col != target:
        rolling = df[col].rolling(6).mean()

        ax.plot(
            range(len(df)),
            rolling,
            color=MAIN_COLOR,
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
            label="6개월 이동평균"
        )

    set_xticks(ax, df)

    ax.set_title(
        col,
        fontsize=12,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=8
    )

    ax.legend(
        facecolor="#2A2A4A",
        edgecolor="#666688",
        labelcolor=TEXT_COLOR,
        fontsize=8,
        loc="best"
    )

fig.suptitle(
    "④ 전체 변수 시계열 비교",
    fontsize=18,
    fontweight="bold",
    color=TITLE_COLOR,
    y=1.01
)

plt.tight_layout()
plt.show()


# =========================================================
# 16. 종료
# =========================================================
print("EDA 완료")
input("엔터 누르면 종료")