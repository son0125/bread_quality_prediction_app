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


# =========================
# Prediction function
# =========================
def predict_target(model_name, target_name, X):
    model, x_scaler, y_scaler = load_model_bundle(model_name, target_name)

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

    mean_value = np.mean(pred_values)

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

    ax.axhline(
        mean_value,
        color="gray",
        linestyle="--",
        linewidth=2
    )

    ax.text(
        len(model_names) - 0.3,
        mean_value,
        f"Mean: {mean_value:.2f}",
        ha="right",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="dimgray"
    )

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

    ax.set_ylabel(info["ylabel"], fontsize=20, fontweight="bold")
    ax.set_title(
        f"Model-wise Predicted {info['title']}",
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
def show_prediction_section(df, target_name):
    info = TARGET_INFO[target_name]

    missing_files = check_target_files(target_name)

    if missing_files:
        st.warning(f"{info['title']} 예측에 필요한 파일이 부족합니다.")
        st.write("누락된 파일:")
        st.write(missing_files)
        return

    feature_columns = load_feature_columns(target_name)

    missing_cols = [col for col in feature_columns if col not in df.columns]

    if missing_cols:
        st.error(f"{info['title']} 예측에 필요한 입력 컬럼이 부족합니다.")
        st.write("누락된 컬럼:")
        st.write(missing_cols)

        st.write("모델이 요구하는 입력 컬럼:")
        st.write(feature_columns)

        return

    X = df[feature_columns].astype(float)

    prediction_summary = []

    for model_name in MODEL_NAMES:
        pred = predict_target(model_name, target_name, X)

        prediction_summary.append({
            "Model": model_name,
            "Mean_Predicted_Value": np.mean(pred)
        })

    summary_df = pd.DataFrame(prediction_summary)

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

    df = pd.read_excel(uploaded_file)

    # 컬럼명을 문자열로 통일
    df.columns = [str(col) for col in df.columns]

    tab1, tab2 = st.tabs(["Hardness Prediction", "Volume Prediction"])

    with tab1:
        show_prediction_section(df, "hardness")

    with tab2:
        show_prediction_section(df, "volume")

else:
    st.info("새 hyperspectral image data Excel 파일을 업로드하세요.")
