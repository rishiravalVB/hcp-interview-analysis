import streamlit as st
import pandas as pd
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import plotly.express as px

# ====== CONFIG ======
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="HCP Interview Analysis", layout="wide")

# ====== FILE UPLOAD ======
st.title("HCP Interview Analysis & Insights Dashboard")
uploaded_file = st.file_uploader("Upload your HCP interview CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Combine comments into one text field
    df["all_comments"] = df[["Comment_1", "Comment_2", "Comment_3"]].astype(str).agg(" ".join, axis=1)

    # ====== CLUSTERING ======
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["all_comments"])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    df["cluster"] = clusterer.fit_predict(X)

    # ====== GENERATE CLUSTER NAMES ======
    cluster_names = {}
    for cluster_id in df["cluster"].unique():
        if cluster_id == -1:
            cluster_names[cluster_id] = "Uncategorized Feedback"
        else:
            sample_texts = " ".join(df[df["cluster"] == cluster_id]["all_comments"].tolist()[:5])
            prompt = f"""
            You are an expert in healthcare market research. 
            Summarize the main theme of the following HCP feedback in 3-5 words for use as a cluster name:

            {sample_texts}

            Respond with ONLY the name, no punctuation except spaces.
            """
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20
                )
                cluster_name = response.choices[0].message.content.strip()
                cluster_names[cluster_id] = f"{cluster_name} (Cluster {cluster_id})"
            except Exception:
                # 🔄 TF-IDF fallback if OpenAI fails
                cluster_terms = vectorizer.get_feature_names_out()
                cluster_vecs = X[df["cluster"] == cluster_id].toarray().sum(axis=0)
                top_terms = [cluster_terms[i] for i in cluster_vecs.argsort()[-3:][::-1]]
                cluster_names[cluster_id] = f"{' '.join(top_terms)} (Cluster {cluster_id})"

    df["cluster_name"] = df["cluster"].map(cluster_names)

    # ====== VISUALIZATIONS ======
    st.subheader("Cluster Distribution")
    fig = px.histogram(df, x="cluster_name", title="Number of HCPs per Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Specialty by Cluster")
    fig2 = px.histogram(df, x="cluster_name", color="Specialty", barmode="group",
                        title="Specialty Distribution per Cluster")
    st.plotly_chart(fig2, use_container_width=True)

    # ====== RECOMMENDATIONS ======
    st.subheader("AI-Generated Recommendations per Cluster")
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster_id]["all_comments"].tolist()
        if cluster_id != -1:
            rec_prompt = f"""
            Based on the following HCP feedback, provide 3 clear, actionable recommendations for the commercial team:
            {cluster_data}
            """
            try:
                rec_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": rec_prompt}],
                    max_tokens=200
                )
                rec_text = rec_response.choices[0].message.content.strip()
            except Exception:
                rec_text = "Error generating recommendations."

            st.markdown(f"**{cluster_names[cluster_id]}**")
            st.write(rec_text)
        else:
            st.markdown("**Uncategorized Feedback**")
            st.write("No specific recommendations — consider reviewing manually.")
