
import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load user credentials
with open("users.json", "r") as f:
    user_db = json.load(f)

st.title("🔐 AI Shopping Recommender")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username in user_db and user_db[username]["password"] == password:
        role = user_db[username]["role"]
        st.success(f"✅ Welcome, {username.title()}! Role: {role}")
        
        df = pd.read_csv("ecommerce_100_users.csv")

        if role == "admin":
            st.subheader("📊 Admin Panel")
            st.dataframe(df)

        elif role == "user":
            st.subheader("🧑 User Profile")

            age = st.number_input("Age", min_value=10, max_value=100)
            gender = st.selectbox("Gender", ["M", "F"])
            browsing = st.text_input("Browsing History (e.g., fashion,shoes)")
            purchase = st.text_input("Purchase History (e.g., dress,heels)")
            location = st.text_input("Location (e.g., NY)")

            if st.button("Get Recommendations"):
                new_user = {
                    'UserID': df['UserID'].max() + 1,
                    'Age': age,
                    'Gender': gender,
                    'Browsing_History': browsing,
                    'Purchase_History': purchase,
                    'Location': location
                }

                df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
                df['combined'] = df['Browsing_History'] + ' ' + df['Purchase_History']

                vectorizer = CountVectorizer().fit_transform(df['combined'])
                similarity_matrix = cosine_similarity(vectorizer)

                user_index = df.index[-1]
                scores = list(enumerate(similarity_matrix[user_index]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]

                st.markdown("### 🎯 Top Recommendations for You")
                for i in scores[:3]:
                    st.write(f"👤 User {df['UserID'][i[0]]} → Browsed: {df['Browsing_History'][i[0]]}, Bought: {df['Purchase_History'][i[0]]}")
    else:
        st.error("❌ Invalid username or password")
