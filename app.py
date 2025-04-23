import streamlit as st
import pandas as pd
import pickle
import os

# Set page config
st.set_page_config(
    page_title="OTD Time Predictor",
    layout="wide",
    page_icon="📦"
)

# Title and banner
st.title("📦 Order to Delivery (OTD) Time Forecasting")
st.image("Dependencies/assets/supply_chain_optimisation.jpg", use_column_width=True)

# Load the model
@st.cache_resource
def load_model():
    with open("Dependencies/voting_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Load data
@st.cache_data
def load_data():
    orders = pd.read_csv("Dependencies/data/olist_orders_dataset.csv")
    customers = pd.read_csv("Dependencies/data/olist_customers_dataset.csv")
    items = pd.read_csv("Dependencies/data/olist_order_items_dataset.csv")
    payments = pd.read_csv("Dependencies/data/olist_order_payments_dataset.csv")
    return orders, customers, items, payments

orders, customers, items, payments = load_data()

# User input form
st.subheader("📝 Enter Order Details")

with st.form("order_form"):
    col1, col2 = st.columns(2)
    with col1:
        freight_value = st.slider("Freight Value (USD)", 0.0, 200.0, 50.0)
        payment_value = st.slider("Total Payment (USD)", 0.0, 300.0, 100.0)
        num_items = st.slider("Number of Items", 1, 10, 3)
        shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Same Day"])

    with col2:
        seller_score = st.slider("Seller Score (1-5)", 1.0, 5.0, 4.2)
        customer_score = st.slider("Customer Score (1-5)", 1.0, 5.0, 4.5)
        delivery_distance = st.slider("Estimated Delivery Distance (km)", 1, 3000, 800)
        customer_location = st.selectbox("Customer Location Type", ["Urban", "Suburban", "Rural"])

    submitted = st.form_submit_button("📈 Predict Delivery Time")

    if submitted:
        # Encode categorical features manually for simplicity
        shipping_map = {"Standard": 1, "Express": 2, "Same Day": 3}
        location_map = {"Urban": 1, "Suburban": 2, "Rural": 3}

        input_df = pd.DataFrame({
            "freight_value": [freight_value],
            "payment_value": [payment_value],
            "num_items": [num_items],
            "shipping_method": [shipping_map[shipping_method]],
            "seller_score": [seller_score],
            "customer_score": [customer_score],
            "delivery_distance": [delivery_distance],
            "customer_location": [location_map[customer_location]]
        })

        prediction = model.predict(input_df)[0]
        st.success(f"✅ Estimated Delivery Time: {prediction:.2f} days")

# Visualizations
st.subheader("📊 Sample Order Data")

if st.checkbox("Show Sample Data"):
    st.write(orders.head())

# Footer
st.markdown("""
---
🚀 This application predicts delivery time based on order details using a trained ML model. 
🔍 Input includes product delivery options, customer location, shipping method, and more.
📦 Deployed via Streamlit and ready for real-world use!
""")
