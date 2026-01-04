import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD ARTIFACTS ----------------
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")
feature_order = joblib.load("feature_order.pkl")
feature_importances = model.feature_importances_  # for plotting

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Financial Distress Prediction",
    layout="wide",
    page_icon="💰"
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("ℹ️ Instructions & Tips")
st.sidebar.write("""
- Fill in your details carefully.  
- Numeric inputs should reflect your actual situation.  
- Click **Predict** to see your financial distress probability.  
- Seek counseling if distress is detected.
""")
st.sidebar.markdown("---")
st.sidebar.header("📌 About")
st.sidebar.write("""
This system predicts financial distress among university students using a **Random Forest model**.  
It helps identify students who might need financial guidance or support.
""")

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align: center;">
    <h1>🎓 Financial Distress Prediction System</h1>
    <p style="font-size:18px;color:gray;">
    Predict financial distress among university students using a Random Forest model.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- USER INPUT ----------------
st.header("📋 Student Information")
st.markdown("Fill out the following details:")

cols = st.columns(2)
user_input = {}

for i, feature in enumerate(feature_order):
    col = cols[i % 2]  # alternate columns
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        user_input[feature] = col.selectbox(f"🔹 {feature}", options)
    else:
        user_input[feature] = col.number_input(f"🔹 {feature}", value=0)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict", type="primary"):
    progress = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.1)
        progress.progress(percent)

    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Keep only model features
    input_df = input_df[selected_features]

    # Predict probability and class
    prob = model.predict_proba(input_df)[0][1]  # probability of distress
    prediction = model.predict(input_df)[0]

    st.markdown("---")

    # ---------------- PROBABILITY GAUGE ----------------
    st.subheader("📊 Financial Distress Probability")
    st.progress(int(prob * 100))
    st.write(f"Predicted Probability of Financial Distress: **{prob*100:.2f}%**")

    # ---------------- PREDICTION CARD ----------------
    if prediction == 1:
        st.markdown(f"""
            <div style="background-color:#ffe6e6;padding:25px;border-radius:12px;border:1px solid #ff4d4d;">
            <h2 style="color:#b40000;">⚠️ Financial Distress Detected</h2>
            <p style="font-size:16px;color:black">It is recommended to seek financial counseling or support to manage your situation effectively.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color:#e6ffe6;padding:25px;border-radius:12px;border:1px solid #33cc33;">
            <h2 style="color:#006600;">✅ No Financial Distress Detected</h2>
            <p style="font-size:16px;color:black">Your financial situation appears stable. Keep managing your finances responsibly!</p>
            </div>
        """, unsafe_allow_html=True)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📈 Feature Importance")
    fi_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
    ax.set_title("Top Factors Contributing to Prediction")
    st.pyplot(fig)
