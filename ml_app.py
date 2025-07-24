# ml_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.datasets import load_iris

def run_ml_app():
    # Load dataset
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['species'] = [iris.target_names[i] for i in iris.target]
    target_names = iris.target_names

    # Sidebar input
    st.sidebar.header("📥 Enter Flower Parameters")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3, 0.1)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Tabs
    tab1, tab2 = st.tabs(["🔮 Species Prediction", "📊 Data Visualization"])

    with tab1:
        st.subheader("🔍 Prediction Result")
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict(input_data)
            predicted_class = target_names[prediction[0]]

            st.success(f"🌼 Predicted Species: **{predicted_class.capitalize()}**")
            st.info(f"""
            **📌 Your Input Details**  
            • Sepal Length: `{sepal_length}` cm  
            • Sepal Width: `{sepal_width}` cm  
            • Petal Length: `{petal_length}` cm  
            • Petal Width: `{petal_width}` cm
            """)
        except FileNotFoundError:
            st.error("❌ The file `model.pkl` was not found. Please ensure the model is in the same directory.")

    with tab2:
        st.subheader("📈 Iris Dataset Distribution (3D View)")
        fig = px.scatter_3d(
            df_iris,
            x='sepal length (cm)',
            y='sepal width (cm)',
            z='petal length (cm)',
            color='species',
            symbol='species',
            title="3D Scatter Plot: Sepal and Petal Dimensions",
            template='plotly_white',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("🔍 **Legend**: Each dot represents an iris flower, colored by its species.")