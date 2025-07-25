import streamlit as st
import pandas as pd
import ralf  # Assuming ralf is installed and importable
import openai  # Ensure openai is installed and importable
from openai import OpenAI
import os
import json
import re



def analyze_problem_type(df, source_col, target_col):
    """Analyze the problem type using GPT-4o-mini based on selected columns."""
    # Take a sample of 5 rows for context
    sample_df = df[[source_col, target_col]].dropna().sample(n=min(200, len(df)), random_state=42)
    sample_text = sample_df.to_csv(index=False)

    prompt = (
        "Given the following pairs of source and target data columns from a dataset, "
        "determine which of the following problem types best describe the task (one or more):\n"
        "- Classification\n"
        "- Summarization\n"
        "- Translation\n"
        "- Code Generation\n"
        "- Reasoning\n"
        "- Instruction Following\n"
        "- Safety & Refusal\n"
        "Only choose from this list. Return a JSON object with two keys: 'types' (a list of the chosen types) and 'reasoning' (a string explaining your reasoning). "
        "Do not include any other text, just the JSON object.\n\n"
        f"Source column: {source_col}\n"
        f"Target column: {target_col}\n"
        f"Sample data:\n{sample_text}"
    )

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
        # Try to extract JSON from the response
        content = response.choices[0].message.content
        # Find the first {...} block in the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # Fallback: try to parse the whole content
        return json.loads(content)
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


st.title("Ralf Model & Dataset Recommender")
# Instantiate Ralf
ralf_instance = ralf.Ralf()
# Take OpenAI API key as input and set environment variable
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key



# Check for GPU and RAM (assuming methods exist; otherwise, add them in ralf.py)
gpu_available = getattr(ralf_instance, "is_gpu_available", lambda: False)()
ram_gb = getattr(ralf_instance, "ram_gb", lambda: "Unknown")
gpu_ram_gb = getattr(ralf_instance, "gpu_ram_gb", lambda: "Unknown")

# Display GPU and RAM information
if gpu_available:
    st.info("GPU is available for processing.")
    st.info(f"GPU RAM (GB): {gpu_ram_gb}")
else:
    st.warning("GPU is not available, using CPU for processing.")
st.info(f"RAM (GB): {ram_gb}")

# Step 1: Upload CSV
# filepath: /workspaces/test_ralf/ralf_app/src/app.py
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



    # Step 3: Run Ralf and display recommendations
    if st.button("Get Recommendations"):
        # Step 2.5: Analyze problem type after both columns are selected
        if source_col and target_col:
            with st.spinner("Analyzing problem type with GPT-4o-mini..."):
                analysis = analyze_problem_type(df, source_col, target_col)
                st.subheader("Problem Type Analysis")
                if isinstance(analysis, dict):
                    st.write("**Types:**", ", ".join(analysis.get("types", [])))
                    st.write("**Reasoning:**")
                    st.write(analysis.get("reasoning", "No reasoning provided."))
                else:
                    st.write(analysis)
        # Replace this with actual ralf usage as per its API
        # Example: results = ralf.recommend(input_csv_file=uploaded_file, source_col=source_col, target_col=target_col)
        # For now, use placeholder results
        results = {
            "llms": ["Llama-2", "Mistral", "Falcon"],
            "golden_dataset": "OpenML Dataset XYZ"
        }
        st.subheader("Recommended Open Source LLMs for Fine-tuning:")
        st.write(results["llms"])
        st.subheader("Recommended Golden Dataset:")
        st.write(results["golden_dataset"])