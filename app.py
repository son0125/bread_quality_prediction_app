import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model


# =========================
# Page setting
# =========================
st.set_page_config(
    page_title="Bread Quality Prediction",
    layout="wide"
)


MODEL_NAMES = [
    "RandomForest",
    "SVM",
    "AdaBoost",
    "GBM",
    "XGBoost",
    "CatBoost",
    "DecisionTree",
    "KNeighbors",
    "LightGBM",
    "MLP",
    "CNN"
]


TARGET_INFO = {
    "hardness": {
        "title": "Hardness",
        "unit": "N",
        "ylabel": "Predicted Hardness (N)",
        "section": "경도 예측 (Hardness Prediction)",
        "chart": "모델별 경도 예측값 비교"
    },
    "volume": {
        "title": "Specific Volume",
        "unit": "mL/g",
        "ylabel": "Predicted Specific Volume (mL/g)",
        "section": "비체적 예측 (Specific Volume Prediction)",
        "chart": "모델별 비체적 예측값 비교"
    }
}


# =========================
# CSS style
# =========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 15px;
        color: #555;
        margin-bottom: 30px;
    }

    .upload-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 4px;
    }

    .upload-desc {
        font-size: 14px;
        color: #555;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 20px;
    }

    .model-card {
        padding: 24px;
        border: 1px solid #d9dee7;
        border-radius: 16px;
        background-color: #f8f9fb;
        text-align: center;
        margin-bottom: 18px;
    }

    .model-name {
        font-size: 17px;
        font-weight: 700;
        color: #333;
        margin-bottom: 12px;
    }

    .pred-value {
        font-size: 44px;
        font-weight: 900;
        color: #111;
    }

    .pred-label {
        font-size: 14px;
        color: #777;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# =========================
# File checking
# =========================
def check_target_files(target_name):
    required_files = [
        f"feature_columns_{target_name}.pkl"
    ]

    for model_name in MODEL_NAMES:
        if model_name in ["MLP", "CNN"]:
            required_files.append(f"{model_name}_{target_name}_model.keras")
        else:
            required_files.append(f"{model_name}_{target_name}_model.pkl")

        required_files.append(f"x_scaler_{model_name}_{target_name}.pkl")
        required_files.append(f"y_scaler_{model_name}_{target_name}.pkl")

    missing_files = [file for file in required_files if not Path(file).exists()]

    return missing_files


# =========================
# Load files
# =========================
@st.cache_resource
def load_feature_columns(target_name):
    feature_columns = joblib.load(f"feature_columns_{target_name}.pkl")
    feature_columns = [str(col) for col in feature_columns]
    return feature_columns


@st.cache_resource
def load_model_bundle(model_name, target_name):
    if model_name in ["MLP", "CNN"]:
        model = load_model(
            f"{model_name}_{target_name}_model.keras",
            compile=False
        )
    else:
        model = joblib.load(f"{model_name}_{target_name}_model.pkl")

    x_scaler = joblib.load(f"x_scaler_{model_name}_{target_name}.pkl")
    y_scaler = joblib.load(f"y_scaler_{model_name}_{target_name}.pkl")

    return model, x_scaler, y_scaler

def validate_hsi_data(df, feature_columns, target_title):
    # 1. 엑셀 파일이 비어 있는 경우
    if df is None or df.empty:
        st.error(
            "올바른 데이터 파일을 업로드하지 않았습니다. "
            "엑셀 파일에 초분광 스펙트럼 데이터가 포함되어 있는지 확인해주세요."
        )
        return None

    # 2. 필요한 HSI 컬럼이 없는 경우
    missing_cols = [col for col in feature_columns if col not in df.columns]

    if missing_cols:
        st.error(
            f"{target_title} 예측을 위한 초분광 스펙트럼 데이터가 올바르게 입력되지 않았습니다. "
            "필요한 HSI 컬럼이 부족합니다."
        )

        with st.expander("누락된 컬럼 확인"):
            st.write(missing_cols)

        return None

    # 필요한 HSI 컬럼만 추출
    X_raw = df[feature_columns].copy()

    # 완전히 빈 문자열 또는 공백을 NaN으로 처리
    X_raw = X_raw.replace(r"^\s*$", np.nan, regex=True)

    # 3. 초분광 데이터에 빈칸이 있는 경우
    if X_raw.isnull().any().any():
        st.error(
            f"{target_title} 예측을 위한 초분광 스펙트럼 데이터에 빈칸이 있습니다. "
            "빈 값을 제거하거나 올바른 숫자 데이터로 입력해주세요."
        )

        wrong_cols = X_raw.columns[X_raw.isnull().any()].tolist()

        with st.expander("빈칸이 있는 컬럼 확인"):
            st.write(wrong_cols)

        return None

    # 4. 숫자가 아닌 문자가 들어 있는 경우
    X_numeric = X_raw.apply(pd.to_numeric, errors="coerce")

    if X_numeric.isnull().any().any():
        st.error(
            f"{target_title} 예측을 위한 초분광 스펙트럼 데이터에 숫자가 아닌 값이 포함되어 있습니다. "
            "초분광 스펙트럼 데이터는 모두 숫자 형태여야 합니다."
        )

        wrong_cols = X_numeric.columns[X_numeric.isnull().any()].tolist()

        with st.expander("문제가 있는 컬럼 확인"):
            st.write(wrong_cols)

        return None

    # 무한대 또는 비정상 값 확인
    if not np.isfinite(X_numeric.values).all():
        st.error(
            f"{target_title} 예측을 위한 초분광 스펙트럼 데이터에 비정상적인 값이 포함되어 있습니다. "
            "무한대 또는 계산 불가능한 값이 있는지 확인해주세요."
        )
        return None

    return X_numeric.astype(float)

# =========================
# Prediction function
# =========================
def predict_target(model_name, target_name, X):
    model, x_scaler, y_scaler = load_model_bundle(model_name, target_name)

    expected_n_features = getattr(x_scaler, "n_features_in_", None)

    if expected_n_features is not None and X.shape[1] != expected_n_features:
        raise ValueError(
            f"모델이 요구하는 입력 변수 개수는 {expected_n_features}개인데, "
            f"업로드된 데이터는 {X.shape[1]}개입니다."
        )
    
    X_scaled = x_scaler.transform(X)

    if model_name == "CNN":
        X_input = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        pred_scaled = model.predict(X_input, verbose=0)

    elif model_name == "MLP":
        pred_scaled = model.predict(X_scaled, verbose=0)

    else:
        pred_scaled = model.predict(X_scaled)

    pred_scaled = np.array(pred_scaled).reshape(-1, 1)
    pred_original = y_scaler.inverse_transform(pred_scaled).ravel()

    return pred_original


# =========================
# Bar chart function
# =========================
def draw_model_bar_chart(summary_df, target_name):
    info = TARGET_INFO[target_name]

    model_names = summary_df["Model"].values
    pred_values = summary_df["Mean_Predicted_Value"].values

    colors = [
        "#4C91D9",
        "#50C878",
        "#F5A623",
        "#E45756",
        "#7B68EE",
        "#20B2AA",
        "#FF8C00",
        "#A0522D",
        "#6A5ACD",
        "#2E8B57",
        "#DC143C"
    ]

    fig, ax = plt.subplots(figsize=(18, 8))

    bars = ax.bar(
        model_names,
        pred_values,
        color=colors[:len(model_names)],
        width=0.65
    )

    # 막대 위 예측값 표시
    for bar, value in zip(bars, pred_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold"
        )

    # y축 여백 추가
    y_min = 0
    y_max = np.max(pred_values) * 1.30

    ax.set_ylim(y_min, y_max)

    ax.set_ylabel(info["ylabel"], fontsize=20, fontweight="bold")
    
    if target_name == "hardness":
        chart_title = "Hardness - Model Prediction"
    elif target_name == "volume":
        chart_title = "Specific Volume - Model Prediction"

    ax.set_title(
        chart_title,
        fontsize=24,
        fontweight="bold",
        pad=20
    )

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(
        [f"{m}\n({v:.2f})" for m, v in zip(model_names, pred_values)],
        rotation=0,
        fontsize=15
    )

    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=15)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()

    return fig

# =========================
# Display prediction section
# =========================
# =========================
# Display prediction section
# =========================
def show_prediction_section(df, target_name):
    info = TARGET_INFO[target_name]

    missing_files = check_target_files(target_name)

    if missing_files:
        st.warning(f"{info['title']} 예측에 필요한 파일이 부족합니다.")
        st.write("누락된 파일:")
        st.write(missing_files)
        return

    feature_columns = load_feature_columns(target_name)

    X = validate_hsi_data(
        df,
        feature_columns,
        info["title"]
    )

    if X is None:
        return

    prediction_summary = []

    for model_name in MODEL_NAMES:
        try:
            pred = predict_target(model_name, target_name, X)

            prediction_summary.append({
                "Model": model_name,
                "Mean_Predicted_Value": np.mean(pred)
            })

        except Exception as e:
            st.error(
                f"{model_name} 모델에서 예측을 수행할 수 없습니다. "
                "업로드한 초분광 스펙트럼 데이터의 컬럼 수, 컬럼명, 데이터 형식이 "
                "학습 데이터와 일치하는지 확인해주세요."
            )

            with st.expander("오류 상세 내용"):
                st.write(str(e))

            return

    summary_df = pd.DataFrame(prediction_summary)

    if summary_df.empty:
        st.error(
            f"{info['title']} 예측 결과를 생성하지 못했습니다. "
            "업로드한 데이터와 모델 파일을 확인해주세요."
        )
        return

    # =========================
    # Prediction table
    # =========================
    st.markdown(
        f"""
        <div class="section-title">
        {info["section"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    value_col_name = f"{info['title']} ({info['unit']})"

    table_df = summary_df.copy()
    table_df = table_df.rename(
        columns={
            "Mean_Predicted_Value": value_col_name
        }
    )

    table_df[value_col_name] = table_df[value_col_name].map(lambda x: f"{x:.4f}")

    st.markdown(
        """
        <style>
        .prediction-table {
            border-collapse: collapse;
            width: 60%;
            margin-top: 10px;
            margin-bottom: 35px;
            font-size: 20px;
        }

        .prediction-table th {
            background-color: #f2f4f8;
            color: #111;
            font-weight: 800;
            text-align: center;
            padding: 16px;
            border-bottom: 2px solid #222;
        }

        .prediction-table td {
            text-align: center;
            padding: 15px;
            border-bottom: 1px solid #d9dee7;
        }

        .prediction-table tr:nth-child(even) {
            background-color: #f8f9fb;
        }

        .prediction-table td:first-child {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        table_df.to_html(
            index=False,
            classes="prediction-table",
            escape=False
        ),
        unsafe_allow_html=True
    )

    fig = draw_model_bar_chart(summary_df, target_name)
    st.pyplot(fig)


# =========================
# Main page
# =========================
st.markdown(
    """
    <div class="main-title">
    올레오젤 초분광 이미지를 활용한 지방 대체 빵의 품질 예측
    </div>
    <div class="subtitle">
    Quality Prediction of Fat-Replaced Breads Using Hyperspectral Imaging of Oleogels
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="upload-title">데이터 파일 업로드</div>
    <div class="upload-desc">
    초분광 스펙트럼 데이터 (224 bands, 936–1716.5 nm)
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "데이터 파일 업로드",
    type=["xlsx", "xls"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    try:
        df = pd.read_excel(uploaded_file)
    except Exception:
        st.error(
            "엑셀 파일을 읽을 수 없습니다. 올바른 .xlsx 또는 .xls 파일을 업로드해주세요."
        )
        st.stop()

    # 완전히 빈 행 제거
    df = df.dropna(how="all")

    if df.empty:
        st.error(
            "업로드한 엑셀 파일에 데이터가 없습니다. "
            "초분광 스펙트럼 데이터가 포함된 파일을 업로드해주세요."
        )
        st.stop()

    # 컬럼명을 문자열로 통일
    df.columns = [str(col) for col in df.columns]
    
    sample_count = len(df)

    if sample_count > 1:
        st.info(
            f"업로드된 데이터는 총 {sample_count}개 샘플입니다. "
            f"결과는 각 샘플의 예측값을 계산한 후 평균값으로 표시됩니다."
        )

    tab1, tab2 = st.tabs(["Hardness Prediction", "Volume Prediction"])
    
    with tab1:
        show_prediction_section(df, "hardness")

    with tab2:
        show_prediction_section(df, "volume")

else:
    st.info("새 hyperspectral image data Excel 파일을 업로드하세요.")
