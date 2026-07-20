import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API = "http://127.0.0.1:8000"

st.title("AI Retail")


# LOGIN
user = st.text_input("User")
pwd = st.text_input("Pass", type="password")

if st.button("Login"):
    res = requests.post(f"{API}/login", json={"username": user, "password": pwd})
    try:
        data = res.json()
    except:
        st.error(res.text)
        st.stop()

    if "token" in data:
        st.session_state["token"] = data["token"]
        st.success("OK")


# MAIN
if "token" in st.session_state:

    token = st.session_state["token"]

    msg = st.text_input("Mesaj")

    if st.button("Send"):

        res = requests.post(
            f"{API}/chat",
            json={"message": msg},
            headers={"Authorization": token}
        )

        try:
            data = res.json()
        except:
            st.error(res.text)
            st.stop()

        st.write(data)

    # 🔥 GRAF LIVE
    res = requests.get(
        f"{API}/products",
        headers={"Authorization": token}
    )

    try:
        products = res.json()
    except:
        st.error(res.text)
        st.stop()

    df = pd.DataFrame(products)

    fig, ax = plt.subplots()
    ax.bar(df["name"], df["stock"])
    st.pyplot(fig)