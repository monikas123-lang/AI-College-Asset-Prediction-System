import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Asset Prediction", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("assets.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS asset_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_name TEXT,
    year INTEGER,
    usage_hours INTEGER,
    maintenance_cost REAL
)
""")
conn.commit()

# ---------------- TITLE ----------------
st.title("Predictive Maintenance System for College IT Assets")
st.write("AI-Based Asset Usage & Cost Prediction")

# ---------------- ADD DATA ----------------
st.subheader("Add Asset Data")

asset = st.text_input("Asset Name")
year = st.number_input("Year", 2015, 2050)
usage = st.number_input("Usage Hours", 0)
cost = st.number_input("Maintenance Cost", 0.0)

if st.button("Add Record"):
    if asset:
        c.execute(
            "INSERT INTO asset_data(asset_name, year, usage_hours, maintenance_cost) VALUES (?,?,?,?)",
            (asset, year, usage, cost)
        )
        conn.commit()
        st.success("Record Added")

# ---------------- VIEW DATA ----------------
st.subheader("Asset Data")
df = pd.read_sql("SELECT * FROM asset_data", conn)

if not df.empty:
    st.dataframe(df, use_container_width=True)

# ---------------- AI PREDICTION ----------------
st.subheader("AI Prediction")

if not df.empty:
    asset_list = df["asset_name"].unique()
    selected_asset = st.selectbox("Select Asset", asset_list)

    asset_df = df[df.asset_name == selected_asset]

    if len(asset_df) >= 2:
        X = asset_df[["year"]]
        y = asset_df["usage_hours"]

        model = LinearRegression()
        model.fit(X, y)

        future_year = st.number_input("Predict for Year", 2024, 2050)

        prediction = model.predict([[future_year]])[0]

        st.success(f"Predicted Usage in {future_year}: {int(prediction)} hours")

        fig, ax = plt.subplots()
        ax.scatter(X, y)
        ax.plot(X, model.predict(X))
        ax.set_xlabel("Year")
        ax.set_ylabel("Usage Hours")
        st.pyplot(fig)
    else:
        st.warning("Add at least 2 years data")
