import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("online_retail.csv")

df = load_data()

# ================= LOAD MODELS =================
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("product_similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

products = sorted(similarity_df.index.tolist())

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation, Product Recommendation & Data Explorer")

# ================= SIDEBAR MENU =================
menu = st.sidebar.radio(
    "Select Module",
    (
        "Product Recommendation",
        "Customer Segmentation",
        "Data Explorer"
    )
)

# =================================================
# 🔹 SIDEBAR – DATASET OVERVIEW (COMPACT)
# =================================================
st.sidebar.header("📊 Dataset Overview")

total_customers = df["CustomerID"].nunique()
total_products = df["Description"].nunique()
total_orders = df["InvoiceNo"].nunique()

# Compact (small) metrics instead of st.metric
st.sidebar.markdown(
    f"""
- 👥 **Customers:** {total_customers}  
- 📦 **Products:** {total_products}  
- 🧾 **Orders:** {total_orders}
"""
)

# ---------- RFM + SEGMENT DISTRIBUTION ----------
@st.cache_data
def compute_segment_distribution(df):
    temp = df.dropna(subset=["CustomerID"]).copy()
    temp["Revenue"] = temp["Quantity"] * temp["UnitPrice"]

    reference_date = pd.to_datetime(temp["InvoiceDate"]).max() + pd.Timedelta(days=1)

    rfm = temp.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - pd.to_datetime(x).max()).days,
        "InvoiceNo": "count",
        "Revenue": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    rfm_scaled = scaler.transform(rfm[["Recency", "Frequency", "Monetary"]])
    rfm["Cluster"] = kmeans.predict(rfm_scaled)

    segment_map = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }

    rfm["Segment"] = rfm["Cluster"].map(segment_map)
    return rfm["Segment"].value_counts()

segment_counts = compute_segment_distribution(df)

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Customer Segments")

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(
    segment_counts.values,
    labels=segment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
st.sidebar.pyplot(fig)

# =================================================
# 1️⃣ PRODUCT RECOMMENDATION
# =================================================
if menu == "Product Recommendation":
    st.header("🔍 Product Recommendation")

    st.write(
        "Search or select a product below. You can **type to search** or **pick from the list**."
    )

    product_name = st.selectbox(
        "Product Name (Type to Search)",
        options=products
    )

    if st.button("Get Recommendations"):
        scores = similarity_df[product_name].sort_values(ascending=False)
        recommendations = scores.iloc[1:6].index.tolist()

        st.success(f"Top 5 products similar to **{product_name}**:")
        for i, prod in enumerate(recommendations, start=1):
            st.write(f"{i}. {prod}")

# =================================================
# 2️⃣ CUSTOMER SEGMENTATION
# =================================================
if menu == "Customer Segmentation":
    st.header("👤 Customer Segmentation")
    st.write("Enter customer RFM values to predict the segment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (days)", min_value=0)
    with col2:
        frequency = st.number_input("Frequency", min_value=0)
    with col3:
        monetary = st.number_input("Monetary Value", min_value=0.0)

    if st.button("Predict Segment"):

        if recency == 0 and frequency == 0 and monetary == 0:
            st.error("❌ Please enter valid customer data before prediction.")
        elif frequency == 0 or monetary == 0:
            st.warning("⚠️ Frequency and Monetary value must be greater than zero.")
        else:
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

            if segment == "High-Value Customer":
                st.info("Frequent purchases, high spending → Loyalty rewards.")
            elif segment == "Regular Customer":
                st.info("Consistent purchases → Cross-selling opportunities.")
            elif segment == "Occasional Customer":
                st.info("Low frequency → Discount campaigns.")
            elif segment == "At-Risk Customer":
                st.warning("Long inactivity → Retention strategies.")

# =================================================
# 3️⃣ DATA EXPLORER
# =================================================
if menu == "Data Explorer":
    st.header("📊 Data Explorer – Online Retail Dataset")

    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        csv,
        "online_retail.csv",
        "text/csv"
    )
