import streamlit as st
import pandas as pd
import ralf  # Assuming ralf is installed and importable
import openai  # Ensure openai is installed and importable
from openai import OpenAI
import os
import json
import re
from ralf import Ralf

# Banner title at the top center
st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50; margin-bottom: 2rem;'>RALF Dashboard</h1>
    """,
    unsafe_allow_html=True
)
# Create sidebar tabs
tab_names = ["Setup", "Recommendation", "Augment", "Lock-In", "Future-Proof"]
selected_tab = st.sidebar.radio("Navigation", tab_names)
ralf = Ralf( HF_TOKEN="abc", OPENAI_API_KEY="abc")

if selected_tab == "Setup":
    st.header("Setup")

    # API Key Inputs
    openai_key = st.text_input("Enter your OPENAI_API_KEY", type="password", key="openai_key")
    gemini_key = st.text_input("Enter your GEMINI_API_KEY", type="password", key="gemini_key")
    hf_token = st.text_input("Enter your HF_TOKEN", type="password", key="hf_token")
   
    show_sysinfo = False
    if st.button("Get System Info"):
        if openai_key:
            ralf = Ralf(OPENAI_API_KEY=openai_key, GEMINI_API_KEY=gemini_key, HF_TOKEN=hf_token)
            show_sysinfo = True
        else:
            st.warning("Please enter your OPENAI_API_KEY.")

    if show_sysinfo:
        # Display Ralf system info
        st.subheader("System Information")
        st.write(f"**GPU Available:** {getattr(ralf, 'gpu_available', 'Unknown')}")
        st.write(f"**GPU RAM (GB):** {getattr(ralf, 'gpu_ram_gb', 'Unknown')}")
        st.write(f"**System RAM (GB):** {getattr(ralf, 'ram_gb', 'Unknown')}")


    uploaded_file = st.file_uploader("Upload your platinum (CSV) dataset", type=["csv"])
    if uploaded_file:
        st.info("File uploaded, reading into DataFrame...")
        with st.spinner("Reading CSV file..."):
            df = pd.read_csv(uploaded_file)
        st.success("File loaded!")
        st.write("Preview of uploaded data:", df.head())

        # Step 2: Select source and target columns
        columns = df.columns.tolist()
        source_col = st.selectbox("Select source column", columns)
        target_col = st.selectbox("Select target column", columns)

        # Store selections in session state for use in other tabs
        st.session_state["df"] = df
        st.session_state["source_col"] = source_col
        st.session_state["target_col"] = target_col
        st.session_state["uploaded_file"] = uploaded_file
        uploaded_file

elif selected_tab == "Recommendation":
    st.header("Recommendation")
    uploaded_file = st.session_state.get("uploaded_file", None)
    if "df" in st.session_state and "source_col" in st.session_state and "target_col" in st.session_state:
        df = st.session_state["df"]
        source_col = st.session_state["source_col"]
        target_col = st.session_state["target_col"]
        csv_file = st.session_state["uploaded_file"]    
        columns = df.columns.tolist()

        source_col, target_col = "source", "target"
        analysis = ralf.analyze_problem_type(df, source_col, target_col)

        # Display the stored results
        display(Markdown(f"# Recommendation Results for {csv_file}:"))

        display(Markdown("## Problem Type Analysis"))
        if isinstance(analysis, dict):
            types = analysis.get('types', [])
            display(Markdown(f"**Types:** {', '.join(types)}"))
            display(Markdown("**Reasoning:**"))
            display(Markdown(analysis.get("reasoning", "No reasoning provided.")))

        llm_recommendations_df, dataset_recommendation_df, analysis_result = ralf.recommend(
            input_csv_file=csv_file,
            source_col=source_col,
            target_col=target_col)

        print("\nRecommended Open Source LLMs for Fine-tuning:")
        display(llm_recommendations_df)

        print("\nRecommended Golden Dataset:")
        display(dataset_recommendation_df)

        print("\nProblem Type Analysis:")
        if isinstance(analysis_result, dict):
            display(Markdown(f"**Types:** {', '.join(analysis_result.get('types', []))}"))
            display(Markdown("**Reasoning:**"))
            display(Markdown(analysis_result.get("reasoning", "No reasoning provided.")))
        else:
            display(Markdown(analysis_result))
    else:
        st.info("Please complete the Setup tab first.")

elif selected_tab == "Augment":
    st.header("Augment")
    st.write("Augmentation features coming soon.")

elif selected_tab == "Lock-In":
    st.header("Lock-In")
    st.write("Lock-In features coming soon.")

elif selected_tab == "Future-Proof":
    st.header("Future-Proof")
    st.write("Future-Proof features coming soon.")