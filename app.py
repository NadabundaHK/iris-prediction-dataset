# app.py

import streamlit as st
import streamlit.components.v1 as stc
from ml_app import run_ml_app

# HTML Header
html_temp = """
    <div style="background-color:#e91e63;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🌸 Iris Flower Classifier App 🌸</h1>
        <h4 style="color:white;text-align:center;">Built with Streamlit & Scikit-Learn</h4>
    </div>
"""

# Deskripsi di halaman Home
desc_temp = """
### Welcome to the Iris Flower Classification App!

This simple web app allows you to classify Iris flower species based on:
- Sepal length & width
- Petal length & width

#### Features:
- 🌼 Machine Learning Prediction (Random Forest)
- 📊 3D Visualization of the Iris dataset

#### Dataset Source:
- `sklearn.datasets.load_iris()`
"""

# Main Function
def main():
    st.set_page_config(page_title="Iris App", page_icon="🌸")
    stc.html(html_temp)

    menu = ["Home", "Predict Species"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("🏠 Home")
        st.markdown(desc_temp)
    elif choice == "Predict Species":
        run_ml_app()

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Powered by Data & ❤️ – Nadabunda Husnul Khotimah</b></p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()