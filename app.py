import streamlit as st
import joblib
from pathlib import Path

from model_wrapper import LSTMWrapper
import joblib
from pathlib import Path

model_path = Path(r"G:\College\6th semester\Minor Project\Fake_News_Model.pkl")
classifier = joblib.load(model_path)


# Predict function using the wrapper
def predict_news(text):
    label, confidence_score = classifier.predict(text)
    return label, confidence_score

# Add Background using CSS
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://source.unsplash.com/1600x900/?news,media");
            background-size: cover;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()

# Streamlit App UI
st.title("📰 LIAR Dataset Fake News Detector (LSTM-based)")
st.write("Enter a news headline or article to check if it's Fake or Real using an LSTM model trained on the LIAR dataset.")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This Fake News Detector uses an LSTM model trained on the LIAR dataset."
    " The input text is tokenized and passed through a neural network to classify it as Fake or Real with a confidence score."
)
st.sidebar.write("🔎 Enter a news article in the main panel to check for credibility.")

# Text input
user_input = st.text_area("Enter News Text:", "")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        result, confidence_score = predict_news(user_input)
        if result == "Fake":
            st.error(f"The news is **{result}** 🛑")
        else:
            st.success(f"The news is **{result}** ✅")
        st.write(f"📊 Confidence Score: **{confidence_score:.2%}**")

# Clear button
if st.button("Clear"):
    st.experimental_rerun()

# Footer
st.caption("🧠 Built using LSTM, trained on the LIAR dataset with Streamlit interface.")
