import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns  # <--- NEW: For plotting

warnings.filterwarnings("ignore")

# --- 1. SET PAGE CONFIG (MUST BE THE FIRST ST COMMAND) ---
st.set_page_config(page_title="Simple ASD Predictor", layout="wide")


# --- 2. Load Model Artifacts ---
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("model/model.joblib")
        scaler = joblib.load("model/scaler.joblib")
        encoders = joblib.load("model/encoders.joblib")
        model_columns = joblib.load("model/model_columns.joblib")
        return model, scaler, encoders, model_columns
    except FileNotFoundError:
        st.error(
            "🔴 **Error:** Model files not found. Run `train_simple_model.py` first."
        )
        return None, None, None, None


model, scaler, encoders, model_columns = load_model_artifacts()


# --- 3. (NEW) Load Raw Data for EDA Visuals ---
@st.cache_data  # <--- NEW: Use cache for data loading
def load_eda_data():
    """
    Loads and preprocesses the raw CSV data for visualization.
    """
    try:
        df = pd.read_csv("dataset/autism_data.csv")

        # We need to re-create the 'Total_AQ_Score' for the plot
        score_cols = [f"A{i}_Score" for i in range(1, 11)]
        df["Total_AQ_Score"] = df[score_cols].sum(axis=1)
        return df
    except FileNotFoundError:
        st.sidebar.error("autism_data.csv not found. Cannot show plots.")
        return None


eda_df = load_eda_data()  # <--- NEW: Load the data


# --- 4. Build the Web Page UI ---
st.title("Simple Autism Screening Dashboard 📊")
st.write("This tool uses a Decision Tree model to make a prediction.")

if model is None:
    st.stop()

# --- 5. (NEW) Sidebar Visualizations ---
if eda_df is not None:
    st.sidebar.title("Data Insights 📈")

    st.sidebar.write(
        "These plots show the patterns in the original data that the model learned from."
    )

    # Plot 1: Total AQ Score Distribution
    st.sidebar.subheader("Total AQ Score by Class")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=eda_df, x="Total_AQ_Score", hue="Class/ASD", kde=True, ax=ax1)
    ax1.set_title("AQ Score is a Strong Predictor")
    st.sidebar.pyplot(fig1)

    # Plot 2: Class Imbalance
    st.sidebar.subheader("Data Imbalance")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=eda_df, x="Class/ASD", ax=ax2)
    ax2.set_title("More 'NO' cases than 'YES'")
    st.sidebar.pyplot(fig2)


# --- 6. Create the Input Form ---
with st.form(key="prediction_form"):

    st.header("Screening Questions (AQ-10)")
    col1, col2 = st.columns(2)
    aq_scores = {}

    def get_binary_answer(label, key):
        return st.selectbox(
            label, (0, 1), format_func=lambda x: "Yes" if x == 1 else "No", key=key
        )

    with col1:
        aq_scores["A1_Score"] = get_binary_answer("A1: Notices small sounds", "a1")
        aq_scores["A2_Score"] = get_binary_answer(
            "A2: Concentrates on whole picture", "a2"
        )
        aq_scores["A3_Score"] = get_binary_answer("A3: Easily notices patterns", "a3")
        aq_scores["A4_Score"] = get_binary_answer(
            "A4: Can get lost in conversation", "a4"
        )
        aq_scores["A5_Score"] = get_binary_answer("A5: Is good at 'chatting'", "a5")

    with col2:
        aq_scores["A6_Score"] = get_binary_answer("A6: Knows if someone is bored", "a6")
        aq_scores["A7_Score"] = get_binary_answer(
            "A7: Doesn't mind routine change", "a7"
        )
        aq_scores["A8_Score"] = get_binary_answer(
            "A8: Easy to 'read between the lines'", "a8"
        )
        aq_scores["A9_Score"] = get_binary_answer("A9: Likes to collect info", "a9")
        aq_scores["A10_Score"] = get_binary_answer(
            "A10: Finds it hard to make new friends", "a10"
        )

    st.header("Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (in years)", min_value=4, max_value=100, value=25)
        gender = st.selectbox(
            "Gender", ("f", "m"), format_func=lambda x: "Female" if x == "f" else "Male"
        )
        jundice = st.selectbox("Jaundice at Birth?", ("no", "yes"))
    with col2:
        austim = st.selectbox("Family Member with ASD?", ("no", "yes"))
        relation = st.selectbox(
            "Who is filling this form?", ("Self", "Parent", "Relative", "Other")
        )

    submit_button = st.form_submit_button(label="Get Prediction")

# --- 7. Handle Form Submission (Logic) ---
if submit_button:
    data = {
        "age": age,
        "gender": gender,
        "jundice": jundice,
        "austim": austim,
        "relation": relation,
    }
    data.update(aq_scores)
    df = pd.DataFrame(data, index=[0])

    # Preprocessing
    score_cols = [f"A{i}_Score" for i in range(1, 11)]
    df["Total_AQ_Score"] = df[score_cols].sum(axis=1)

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(np.ravel(df[col]))

    df_reordered = pd.DataFrame(columns=model_columns, index=[0])
    for col in model_columns:
        if col in df.columns:
            df_reordered[col] = df[col]
        else:
            df_reordered[col] = 0

    df_scaled = scaler.transform(df_reordered)

    # Prediction
    with st.spinner("Analyzing..."):
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)

    result_label = "YES" if prediction[0] == 1 else "NO"
    confidence = float(probability[0][prediction[0]]) * 100

    st.subheader("Prediction Result")
    if result_label == "YES":
        st.error(f"**Prediction: ASD ({result_label})**")
    else:
        st.success(f"**Prediction: No ASD ({result_label})**")
    st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")

# --- 8. EXPLANATION SECTION ---
st.subheader("How the Model Makes a Decision")
st.write(
    """
This model is a 'Decision Tree'. It asks a series of simple 'if-then' 
questions to arrive at a decision. Below is the flowchart the model uses.
"""
)

with st.expander("Click to see the Model's Flowchart"):
    fig, ax = plt.subplots(figsize=(25, 15))
    plot_tree(
        model,
        feature_names=model_columns,
        class_names=["NO", "YES"],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=3,
    )
    st.pyplot(fig)
