import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ------->> Here LOAD MODELS (ORDER FIXED)
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("product_similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

# ------------>> PRODUCT LIST (AFTER LOAD)
products = similarity_df.index.tolist()
products.sort()

# --------------->> PAGE CONFIG
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.title("Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation")

# --------------->> SIDEBAR
menu = st.sidebar.radio(
    "Select Module",
    ("Product Recommendation", "Customer Segmentation")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Customer Segment Distribution")

# Dummy distribution (dashboard purpose)
segment_data = {
    "High-Value": 30,
    "Regular": 45,
    "Occasional": 15,
    "At-Risk": 10
}

fig, ax = plt.subplots()
ax.pie(
    segment_data.values(),
    labels=segment_data.keys(),
    autopct='%1.1f%%'
)
ax.set_title("Customer Segments")
st.sidebar.pyplot(fig)

# -------------->> PRODUCT RECOMMENDATION 
if menu == "Product Recommendation":
    st.header("Product Recommendation")

    product_name = st.selectbox(
        "Select a Product",
        options=products
    )

    if st.button("Get Recommendations"):
        scores = similarity_df[product_name].sort_values(ascending=False)
        recommendations = scores.iloc[1:6].index.tolist()

        st.success("Top 5 Similar Products:")
        for prod in recommendations:
            st.write("•", prod)

# ---------------->> CUSTOMER SEGMENTATION 
if menu == "Customer Segmentation":
    st.header("Customer Segmentation")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency", min_value=0)
    monetary = st.number_input("Monetary Value", min_value=0.0)

    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]

        segment_map = {
            0: "High-Value Customer",
            1: "Regular Customer",
            2: "Occasional Customer",
            3: "At-Risk Customer"
        }

        segment = segment_map.get(cluster, "Unknown")

        st.success(f"Customer Segment: {segment}")

        # ------>> EXPLANATION BOX 
        if segment == "High-Value Customer":
            st.info("This customer purchases frequently, spends more, and has bought recently.")
        elif segment == "Regular Customer":
            st.info("This customer shows consistent purchasing behavior with moderate spending.")
        elif segment == "Occasional Customer":
            st.info("This customer purchases infrequently and contributes lower revenue.")
        elif segment == "At-Risk Customer":
            st.warning("This customer has not purchased for a long time and may churn.")
