import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from xgboost_2026_jan_feb import (
    SCENARIO_2026_JAN,
    SCENARIO_2026_FEB,
    SCENARIO_2026_MAR,
    train_xgb_and_predict_2026,
)

st.set_page_config(
    page_title="Air Freight Index 예측",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Air Freight Index 예측")

FEATURE_LABELS = {
    "wti_price": "WTI 유가",
    "brent_price": "Brent 유가",
    "dxy_index": "달러 인덱스",
    "vix": "VIX 지수",
    "GSCPI": "GSCPI",
    "BDI": "BDI",
    "GPR": "GPR",
    "GPRH": "GPRH",
    "GDP": "GDP",
}

ACTUAL_2026 = {
    "2026-01": 177.227,
    "2026-02": 175.010,
    "2026-03": 176.501,
}


# =========================================================
# 모델 로드 / 예측 헬퍼
# =========================================================
@st.cache_resource
def load_model(with_gdp):
    model, df, features, result = train_xgb_and_predict_2026(with_gdp=with_gdp)
    return model, df, features


def make_input_df(features, values):
    return pd.DataFrame([{feature: values[feature] for feature in features}])


def default_value(feature):
    if feature in SCENARIO_2026_MAR:
        return float(SCENARIO_2026_MAR[feature])
    if feature in SCENARIO_2026_FEB:
        return float(SCENARIO_2026_FEB[feature])
    if feature in SCENARIO_2026_JAN:
        return float(SCENARIO_2026_JAN[feature])
    return 0.0


def predict_2026_jan_to_mar(model, features):
    rows = []

    scenarios = {
        "2026-01": SCENARIO_2026_JAN,
        "2026-02": SCENARIO_2026_FEB,
        "2026-03": SCENARIO_2026_MAR,
    }

    for month, scenario in scenarios.items():
        values = {}

        for feature in features:
            values[feature] = float(scenario.get(feature, default_value(feature)))

        input_df = make_input_df(features, values)
        pred = float(model.predict(input_df)[0])

        rows.append(
            {
                "Date": pd.to_datetime(month + "-01"),
                "Actual": ACTUAL_2026[month],
                "Predicted": pred,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 오차 계산 헬퍼
# =========================================================
def calc_error_metrics(actual, predicted):
    """실제값/예측값으로 오차·오차율·정확도 계산. 실제값이 None이면 None 반환."""
    if actual is None or pd.isna(actual):
        return {
            "Error": None,
            "Abs_Error": None,
            "Error_Rate_%": None,
            "Accuracy_%": None,
        }
    error = actual - predicted
    abs_error = abs(error)
    error_rate = abs_error / actual * 100 if actual != 0 else None
    accuracy = 100 - error_rate if error_rate is not None else None
    return {
        "Error": error,
        "Abs_Error": abs_error,
        "Error_Rate_%": error_rate,
        "Accuracy_%": accuracy,
    }


# =========================================================
# 차트 그리기
# =========================================================
def draw_chart(history_df, base_pred_df, user_pred_df=None):
    df = history_df.copy()
    df["Date"] = df["YearMonth"].dt.to_timestamp()
    df = df.sort_values("Date")

    fig = go.Figure()

    # 1. 과거 실제값 (짙은 파란색)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Air_Freight_Index"],
            mode="lines",
            name="과거 실제값",
            line=dict(color="#1f77b4", width=2), # Plotly 기본 파란색
        )
    )

    # 2. 과거 모델 예측값 (주황색)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Model_Predicted"],
            mode="lines",
            name="과거 모델 예측값",
            line=dict(dash="dash", color="#FF6B35", width=2),
        )
    )

    # 3. 2026 1~3월 실제값 라인 (짙은 파란색)
    last_actual = pd.DataFrame({
        "Date": [df["Date"].iloc[-1]],
        "Actual": [df["Air_Freight_Index"].iloc[-1]],
    })

    actual_line = pd.concat(
        [last_actual, base_pred_df[["Date", "Actual"]]],
        ignore_index=True,
    )

    fig.add_trace(
        go.Scatter(
            x=actual_line["Date"],
            y=actual_line["Actual"],
            mode="lines+markers+text",
            name="2026 실제값",
            text=[""] + [f"{v:.2f}" for v in base_pred_df["Actual"]],
            textposition="top center",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=10, color="#1f77b4", line=dict(width=2, color="white")),
            textfont=dict(color="#1f77b4", size=11, family="Arial Black"), # 글씨도 파란색으로 통일
        )
    )

    # 4. 2026 1~3월 모델 예측값 라인 (주황색)
    last_pred = pd.DataFrame({
        "Date": [df["Date"].iloc[-1]],
        "Predicted": [df["Model_Predicted"].iloc[-1]],
    })

    pred_line = pd.concat(
        [last_pred, base_pred_df[["Date", "Predicted"]]],
        ignore_index=True,
    )

    fig.add_trace(
        go.Scatter(
            x=pred_line["Date"],
            y=pred_line["Predicted"],
            mode="lines+markers+text",
            name="모델 예측값",
            text=[""] + [f"{v:.2f}" for v in base_pred_df["Predicted"]],
            textposition="bottom center",
            line=dict(color="#FF6B35", width=2),
            marker=dict(size=10, color="#FF6B35", line=dict(width=2, color="white")),
            textfont=dict(color="#D84315", size=11, family="Arial Black"), # 주황색 텍스트
        )
    )

    # 1~3월 오차선 + 어노테이션
    for _, row in base_pred_df.iterrows():
        actual = row["Actual"]
        pred = row["Predicted"]
        date = row["Date"]
        if actual is None or pd.isna(actual):
            continue
        # 오차선 (점선 - 어두운 회색으로 변경하여 대비감 상승)
        fig.add_trace(
            go.Scatter(
                x=[date, date],
                y=[actual, pred],
                mode="lines",
                line=dict(color="#888888", width=1.5, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ── 사용자 예측값 (실제값 입력했으면 함께 표시) ──
    if user_pred_df is not None and not user_pred_df.empty:
        # 사용자 예측값 라인
        anchor_pred = pd.DataFrame({
            "Date": [base_pred_df["Date"].iloc[-1]],
            "User_Predicted": [base_pred_df["Predicted"].iloc[-1]],
        })
        user_line = pd.concat(
            [anchor_pred, user_pred_df[["Date", "User_Predicted"]]],
            ignore_index=True,
        )
        fig.add_trace(
            go.Scatter(
                x=user_line["Date"],
                y=user_line["User_Predicted"],
                mode="lines+markers+text",
                name="사용자 예측값",
                text=[""] + [f"{v:.2f}" for v in user_pred_df["User_Predicted"]],
                textposition="bottom center",
                line=dict(dash="dash", color="#FF8A65", width=2), # 주황색 톤 유지
                marker=dict(size=11, color="#FF8A65", symbol="diamond",
                           line=dict(width=2, color="white")),
                textfont=dict(color="#D84315", size=11),
            )
        )

        # 5. 사용자 실제값 (짙은 파란색)
        actual_rows = user_pred_df[user_pred_df["User_Actual"].notna()].copy()
        if not actual_rows.empty:
            anchor_actual = pd.DataFrame({
                "Date": [base_pred_df["Date"].iloc[-1]],
                "User_Actual": [base_pred_df["Actual"].iloc[-1]],
            })
            actual_user_line = pd.concat(
                [anchor_actual, actual_rows[["Date", "User_Actual"]]],
                ignore_index=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=actual_user_line["Date"],
                    y=actual_user_line["User_Actual"],
                    mode="lines+markers+text",
                    name="사용자 실제값",
                    text=[""] + [f"{v:.2f}" for v in actual_rows["User_Actual"]],
                    textposition="top center",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=11, color="#1f77b4", symbol="circle",
                               line=dict(width=2, color="white")),
                    textfont=dict(color="#1f77b4", size=11, family="Arial Black"),
                )
            )

            # 사용자 입력값에 대한 오차선
            for _, urow in actual_rows.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[urow["Date"], urow["Date"]],
                        y=[urow["User_Actual"], urow["User_Predicted"]],
                        mode="lines",
                        line=dict(color="#888888", width=1.5, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # 2026-01 기준선
    fig.add_vline(
        x=base_pred_df["Date"].iloc[0],
        line_width=1.5,
        line_dash="solid",
        line_color="#444444",
        opacity=0.6,
    )

    fig.update_layout(
        title="Air Freight Index 예측 차트",
        xaxis_title="날짜",
        yaxis_title="Air Freight Index",
        template="plotly_white",  # 흰색 배경으로 변경
        height=650,
        margin=dict(l=30, r=30, t=60, b=30),
        hovermode="x unified",
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)", # 흰색 반투명 범례 배경
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
    )

    return fig


# =========================================================
# 사이드바 — 모델 선택
# =========================================================
with st.sidebar:
    st.header("모델 선택")
    model_option = st.radio(
        "사용할 모델",
        ["GDP 미포함 모델", "GDP 포함 모델"],
        index=1,
    )
    st.divider()
    if st.button("🗑️ 사용자 예측 기록 초기화"):
        for k in list(st.session_state.keys()):
            if k.startswith("user_predictions"):
                st.session_state[k] = []
        st.success("초기화 완료")
        st.rerun()

with_gdp = model_option == "GDP 포함 모델"

model, history_df, features = load_model(with_gdp)

# session_state 키는 모델별로 분리
prediction_key = "user_predictions_gdp" if with_gdp else "user_predictions_no_gdp"

if prediction_key not in st.session_state:
    st.session_state[prediction_key] = []

base_pred_df = predict_2026_jan_to_mar(model, features)


# =========================================================
# 입력 폼
# =========================================================
st.subheader("사용자 입력")

with st.form("prediction_form"):
    col_date, col_actual = st.columns([1, 1])

    with col_date:
        input_date = st.date_input(
            "예측 날짜",
            value=pd.to_datetime("2026-05-01"),
        )

    with col_actual:
        actual_input = st.text_input(
            "실제 항공 운임 지수 (선택, 비워두면 예측값만 표시)",
            value="",
            placeholder="예: 175.5",
            help="해당 월의 실제 Air Freight Index 값을 알고 있으면 입력하세요. "
                 "비워두면 미래 예측 모드로 작동합니다.",
        )

    st.markdown("##### 피처 입력값")
    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            values[feature] = st.number_input(
                FEATURE_LABELS.get(feature, feature),
                value=default_value(feature),
                format="%.4f",
            )

    submitted = st.form_submit_button("🔍 예측하기", use_container_width=True)


# =========================================================
# 예측 실행
# =========================================================
if submitted:
    input_df = make_input_df(features, values)
    prediction = float(model.predict(input_df)[0])

    # 실제값 파싱
    user_actual = None
    if actual_input.strip():
        try:
            user_actual = float(actual_input.strip())
        except ValueError:
            st.warning("⚠️ 실제값은 숫자만 입력하세요. 실제값을 무시하고 예측값만 표시합니다.")
            user_actual = None

    # 오차 계산
    metrics = calc_error_metrics(user_actual, prediction)

    new_row = {
        "Date": pd.to_datetime(input_date),
        "User_Predicted": prediction,
        "User_Actual": user_actual,
        "Error": metrics["Error"],
        "Abs_Error": metrics["Abs_Error"],
        "Error_Rate_%": metrics["Error_Rate_%"],
        "Accuracy_%": metrics["Accuracy_%"],
    }

    # 같은 날짜는 덮어쓰기
    st.session_state[prediction_key] = [
        row for row in st.session_state[prediction_key]
        if row["Date"] != pd.to_datetime(input_date)
    ]
    st.session_state[prediction_key].append(new_row)
    st.session_state[prediction_key] = sorted(
        st.session_state[prediction_key],
        key=lambda x: x["Date"],
    )

    # 결과 메트릭 표시
    if user_actual is not None:
        st.success(f"✅ 예측 완료 — 실제값과 비교 분석")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("실제값", f"{user_actual:.3f}")
        m2.metric("예측값", f"{prediction:.3f}")
        m3.metric("절대오차", f"{metrics['Abs_Error']:.3f}")
        m4.metric("오차율", f"{metrics['Error_Rate_%']:.2f}%")
        m5.metric("정확도", f"{metrics['Accuracy_%']:.2f}%")
    else:
        st.success(f"✅ 예측 완료 — 미래 예측 모드")
        m1, m2 = st.columns(2)
        m1.metric("예측 날짜", input_date.strftime("%Y-%m"))
        m2.metric("예측값", f"{prediction:.3f}")


user_pred_df = pd.DataFrame(st.session_state[prediction_key])


# =========================================================
# 차트
# =========================================================
st.subheader("예측 차트")

fig = draw_chart(
    history_df=history_df,
    base_pred_df=base_pred_df,
    user_pred_df=user_pred_df,
)

st.plotly_chart(fig, width="stretch")


# =========================================================
# 검증 결과 표 (1~3월 + 사용자 입력)
# =========================================================
st.subheader("📊 검증 결과")

# 1~3월 베이스 결과
base_table = base_pred_df.copy()
base_table["Error"] = base_table["Actual"] - base_table["Predicted"]
base_table["Abs_Error"] = base_table["Error"].abs()
base_table["Error_Rate_%"] = base_table["Abs_Error"] / base_table["Actual"] * 100
base_table["Accuracy_%"] = 100 - base_table["Error_Rate_%"]
base_table["Source"] = "기본 (2026 1~3월)"
base_table["Date_Str"] = base_table["Date"].dt.strftime("%Y-%m")

display_cols = ["Source", "Date_Str", "Actual", "Predicted",
                "Abs_Error", "Error_Rate_%", "Accuracy_%"]

base_show = base_table[display_cols].rename(columns={
    "Date_Str": "날짜",
    "Actual": "실제값",
    "Predicted": "예측값",
    "Abs_Error": "절대오차",
    "Error_Rate_%": "오차율(%)",
    "Accuracy_%": "정확도(%)",
    "Source": "구분",
})

# 사용자 입력 결과
if not user_pred_df.empty:
    user_table = user_pred_df.copy()
    user_table["Date_Str"] = user_table["Date"].dt.strftime("%Y-%m")
    user_table["Source"] = "사용자 입력"
    user_table_renamed = user_table.rename(columns={
        "Date_Str": "날짜",
        "User_Actual": "실제값",
        "User_Predicted": "예측값",
        "Abs_Error": "절대오차",
        "Error_Rate_%": "오차율(%)",
        "Accuracy_%": "정확도(%)",
        "Source": "구분",
    })
    user_show = user_table_renamed[
        ["구분", "날짜", "실제값", "예측값", "절대오차", "오차율(%)", "정확도(%)"]
    ]
    combined = pd.concat([base_show, user_show], ignore_index=True)
else:
    combined = base_show

# 숫자 포맷팅
def format_num(v, decimals=3):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{decimals}f}"

styled = combined.copy()
for col in ["실제값", "예측값", "절대오차"]:
    styled[col] = styled[col].apply(lambda v: format_num(v, 3))
for col in ["오차율(%)", "정확도(%)"]:
    styled[col] = styled[col].apply(lambda v: format_num(v, 2))

st.dataframe(styled, width="stretch", hide_index=True)

# 사용자가 입력한 실제값 기반 평균 정확도
if not user_pred_df.empty:
    valid_user = user_pred_df.dropna(subset=["Accuracy_%"])
    if not valid_user.empty:
        st.markdown("##### 📈 사용자 검증 평균")
        avg1, avg2, avg3 = st.columns(3)
        avg1.metric("평균 절대오차", f"{valid_user['Abs_Error'].mean():.3f}")
        avg2.metric("평균 오차율", f"{valid_user['Error_Rate_%'].mean():.2f}%")
        avg3.metric("평균 정확도", f"{valid_user['Accuracy_%'].mean():.2f}%")
