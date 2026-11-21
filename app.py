import joblib
import pandas as pd
import streamlit as st

# 1. Load model artifact
artifact = joblib.load("house_price_model.pkl")
model = artifact["pipeline"]
columns = artifact["columns"]
metrics_df = artifact["metrics"]
feat_imp_df = artifact["feature_importance"]
best_model_name = artifact["best_model_name"]
best_rmse = metrics_df.iloc[0]["RMSE"]

st.set_page_config(page_title="Real Estate Valuation Engine", layout="wide")
st.title("Real Estate Valuation & Analytics Engine")
st.caption(f"Powered by **{best_model_name}** | Test RMSE: ±{best_rmse} (10k NTD/Ping)")

tab1, tab2 = st.tabs(["Price Predictor & Map", "Model Benchmarks & Insights"])

with tab1:
    col_input, col_map = st.columns([1.2, 1])

    with col_input:
        st.subheader("Property Parameters")
        c1, c2 = st.columns(2)
        with c1:
            trans_date = st.number_input("Transaction Date", min_value=2012.0, max_value=2014.0, value=2013.25, step=0.05)
            house_age = st.slider("House Age (Years)", min_value=0.0, max_value=45.0, value=10.0, step=0.5)
            mrt_distance = st.number_input("Distance to Nearest MRT (m)", min_value=0.0, max_value=6500.0, value=500.0, step=50.0)
        with c2:
            stores = st.slider("Nearby Convenience Stores", min_value=0, max_value=10, value=5)
            latitude = st.number_input("Latitude", min_value=24.93, max_value=25.02, value=24.97000, step=0.005, format="%.5f")
            longitude = st.number_input("Longitude", min_value=121.47, max_value=121.57, value=121.54000, step=0.005, format="%.5f")

        input_data = pd.DataFrame(
            [[trans_date, house_age, mrt_distance, stores, latitude, longitude]],
            columns=columns
        )

        if st.button("Predict Property Value", type="primary", use_container_width=True):
            pred = model.predict(input_data)[0]
            low = max(0, pred - best_rmse)
            high = pred + best_rmse
            st.success(f"Estimated Unit Price: **{pred:.2f}** (10,000 NTD/Ping)")
            st.info(f"Expected Confidence Range (±1 RMSE): **{low:.2f} – {high:.2f}**")

    with col_map:
        st.subheader("Selected Property Location")
        map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_df, zoom=13)

with tab2:
    st.subheader("Model Benchmark Comparison")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Top Price Drivers (Feature Importance)")
    st.bar_chart(feat_imp_df.set_index("Feature"))