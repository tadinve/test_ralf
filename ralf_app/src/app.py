import streamlit as st
import pandas as pd
from ralf import Ralf
import os
# import torch
import psutil
import humanize
import json

st.set_page_config(
    page_title="RALF - Resource-Aware LLM Finetuning",
    page_icon="🤖",
    layout="wide"
)

# ---------------------- STYLES ----------------------
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #DFF2BF;
        color: #4F8A10;
    }
    .error-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #FFE8E6;
        color: #D8000C;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SYSTEM INFO ----------------------
def get_system_info():
#    gpu_available = torch.cuda.is_available()
#    gpu_info = f"{torch.cuda.get_device_name(0)}" if gpu_available else "No GPU"
#    gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB" if gpu_available else "N/A"
    gpu_available = False
    gpu_info = "N/A"
    gpu_memory = "N/A"
    ram = humanize.naturalsize(psutil.virtual_memory().total)
    return {
        "GPU Available": "✅ Yes" if gpu_available else "❌ No",
        "GPU Model": gpu_info,
        "GPU Memory": gpu_memory,
        "System RAM": ram
    }

# ---------------------- MAIN ----------------------
def main():
    st.title("🤖 RALF - Resource-Aware LLM Finetuning")

    # Sidebar
    with st.sidebar:
        st.markdown("## 🖥️ System Information")
        sys_info = get_system_info()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="GPU", value=sys_info["GPU Available"])
            st.metric(label="GPU Memory", value=sys_info["GPU Memory"])
        with col2:
            st.metric(label="GPU Model", value=sys_info["GPU Model"])
            st.metric(label="System RAM", value=sys_info["System RAM"])

        st.markdown("---")
        st.header("API Configuration")
        hf_token = st.text_input("Hugging Face Token", type="password")
        api_choice = st.radio("Choose API for Recommendations", ["OpenAI", "Gemini"])
        if api_choice == "OpenAI":
            openai_key = st.text_input("OpenAI API Key", type="password")
            gemini_key = None
        else:
            gemini_key = st.text_input("Gemini API Key", type="password")
            openai_key = None

        if st.button("Validate Credentials"):
            if not hf_token:
                st.error("Please provide a Hugging Face token")
            elif api_choice == "OpenAI" and not openai_key:
                st.error("Please provide an OpenAI API key")
            elif api_choice == "Gemini" and not gemini_key:
                st.error("Please provide a Gemini API key")
            else:
                try:
                    ralf = Ralf(HF_TOKEN=hf_token,
                                OPENAI_API_KEY=openai_key,
                                GEMINI_API_KEY=gemini_key)
                    st.success("Credentials validated successfully!")
                    st.session_state['credentials_valid'] = True
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                    st.session_state['credentials_valid'] = False

        st.markdown("---")
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    st.error("The uploaded CSV file is empty")
                else:
                    st.write(f"File size: {uploaded_file.size/1024:.2f} KB")
                    st.write(f"Number of rows: {len(df)}")
                    st.write(f"Number of columns: {len(df.columns)}")
                    st.session_state['df'] = df
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # Tabs
    tabs = st.tabs(["Analysis", "Recommendation", "Augmentation", "Lustration", "Futureproofing"])

    # ---------------------- ANALYSIS TAB ----------------------
    with tabs[0]:
        st.header("Dataset Analysis")
        if 'df' in st.session_state:
            st.subheader("Data Preview")
            st.dataframe(st.session_state['df'].head())

            col1, col2 = st.columns(2)
            with col1:
                st.session_state['source_col'] = st.selectbox(
                    "Select source column",
                    st.session_state['df'].columns
                )
            with col2:
                st.session_state['target_col'] = st.selectbox(
                    "Select target column",
                    [col for col in st.session_state['df'].columns if col != st.session_state['source_col']]
                )

            # Data cleaning (moved from Lustration tab)
            st.subheader("Data Cleaning")
            remove_nulls = st.checkbox("Remove rows with missing values")
            lowercase_text = st.checkbox("Convert text to lowercase")
            remove_duplicates = st.checkbox("Remove duplicate rows")

            if st.button("Clean Data"):
                try:
                    cleaned_df = st.session_state['df'].copy()
                    if remove_nulls:
                        cleaned_df.dropna(inplace=True)
                    if lowercase_text:
                        cleaned_df[st.session_state['source_col']] = cleaned_df[st.session_state['source_col']].str.lower()
                        cleaned_df[st.session_state['target_col']] = cleaned_df[st.session_state['target_col']].str.lower()
                    if remove_duplicates:
                        cleaned_df.drop_duplicates(inplace=True)
                    st.session_state['df'] = cleaned_df
                    st.success("Data cleaned successfully!")
                    st.dataframe(cleaned_df.head())
                except Exception as e:
                    st.error(f"Error during data cleaning: {str(e)}")
        else:
            st.info("Please upload your dataset in the sidebar first.")

    # ---------------------- RECOMMENDATION TAB ----------------------
    with tabs[1]:
        st.header("Model Recommendation")
        if 'df' in st.session_state and 'credentials_valid' in st.session_state:
            if st.button("Analyze Dataset"):
                with st.spinner("Analyzing dataset..."):
                    try:
                        temp_csv_path = "temp_dataset.csv"
                        st.session_state['df'].to_csv(temp_csv_path, index=False)
                        ralf = Ralf(HF_TOKEN=hf_token,
                                    OPENAI_API_KEY=openai_key,
                                    GEMINI_API_KEY=gemini_key)
                        llm_df, dataset_df, analysis = ralf.recommend(
                            temp_csv_path,
                            st.session_state['source_col'],
                            st.session_state['target_col']
                        )
                        os.remove(temp_csv_path)
                        st.session_state['llm_df'] = llm_df
                        st.session_state['dataset_df'] = dataset_df
                        st.session_state['analysis'] = analysis
                        st.success("Analysis complete! Check model recommendations below.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

        if 'analysis' in st.session_state:
            st.subheader("🧠 Problem Analysis")
            raw_analysis = st.session_state['analysis']
            if isinstance(raw_analysis, str):
                try:
                    analysis = json.loads(raw_analysis)
                except json.JSONDecodeError:
                    analysis = {"reasoning": raw_analysis}
            else:
                analysis = raw_analysis
            st.markdown(f"**📝 Task Type:** `{analysis.get('types', ['Not specified'])[0]}`")
            st.info(analysis.get("reasoning", "No reasoning provided"))

        if 'llm_df' in st.session_state and not st.session_state['llm_df'].empty:
            st.subheader("Recommended Models")
            llm_df = st.session_state['llm_df'].copy()
            if "Name" in llm_df.columns and "Hugging Face URL" in llm_df.columns:
                llm_df["Name"] = llm_df.apply(
                    lambda row: f'<a href="{row["Hugging Face URL"]}" target="_blank">{row["Name"]}</a>'
                    if pd.notnull(row["Hugging Face URL"]) and pd.notnull(row["Name"]) else row["Name"], axis=1
                )
            llm_df.insert(0, "S.No", range(1, len(llm_df) + 1))
            display_df = llm_df[["S.No", "Name", "Parameters", "Description"]]
            st.markdown(llm_df.to_html(escape=False, index=False), unsafe_allow_html=True)


    # Augmentation Tab
    with tabs[2]:
        st.header("Data Augmentation")
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.subheader("Original Dataset")
            st.dataframe(df.head())
            st.markdown("### Augmentation Options")
            aug_type = st.selectbox(
                "Select augmentation type",
                ["Synonym Replacement", "Back Translation", "Random Swap"])
            num_aug = st.slider("Number of augmented examples per orginal", 1, 5, 2)
            if st.button("Apply Augmentation"):
                try:
                    with st.spinner("Applying augmentation..."):
                        augmented_df = df.copy()
                        augmented_df["augmented_text"] = df[st.session_state['source_col']] + " (augmented)"
                        st.session_state['augmented_df'] = augmented_df
                        st.success("Augmentation applied successfully!")
                        st.dataframe(augmented_df.head())
                except Exception as e:
                    st.error(f"Error during augmentation: {str(e)}")
            else:
                st.info("Please upload and analyze your dataset first to apply augmentation.")
    with tabs[3]:
        st.header("Data Lustration(Cleaning & Preprocessing)")
        #if 'df' in st.session_state:
            #df = st.session_state['df']
            #st.write("Original Dataset:")
            #st.dataframe(df.head())
            #remove_nulls = st.checkbox("Remove rows with missing values")
            #lowercase_text = st.checkbox("Convert text to lowercase")
            #remove_duplicates = st.checkbox("Remove duplicate rows")
            #if st.button("Clean Data"):
                #try:
                    #cleaned_df = df.copy()
                    #if remove_nulls:
                        #cleaned_df.dropna(inplace=True)
                    #if lowercase_text:
                        #cleaned_df[st.session_state['source_col']] = cleaned_df[st.session_state['source_col']].str.lower()
                        #cleaned_df[st.session_state['target_col']] = cleaned_df[st.session_state['target_col']].str.lower()
                    #if remove_duplicates:
                        #cleaned_df.drop_duplicates(inplace=True)
                    #st.session_state['cleaned_df'] = cleaned_df
                    #st.success("Data cleaned successfully!")
                    #st.dataframe(cleaned_df.head())
                #except Exception as e:
                    #st.error(f"Error during data cleaning: {str(e)}")
            #else:
                #st.info("Please upload and analyze your dataset first to apply lustration.")



    # Training Tab
    with tabs[4]:
        st.header("Model Futureproofing")
        if 'selected_models' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Number of epochs", min_value=1, value=3)
                batch_size = st.number_input("Batch size", min_value=1, value=16)
                st.session_state['epochs'] = epochs
                st.session_state['batch_size'] = batch_size
            with col2:
                learning_rate = st.number_input("Learning rate", min_value=0.0, value=2e-5)
                output_dir = st.text_input("Output directory", value="./results")
                st.session_state['learning_rate'] = learning_rate
                st.session_state['output_dir'] = output_dir

            if st.button("Start Training"):
                try:
                    with st.spinner("Training in progress..."):
                        progress_bar = st.progress(0)
                        ralf = Ralf(HF_TOKEN=hf_token, 
                                  OPENAI_API_KEY=openai_key, 
                                  GEMINI_API_KEY=gemini_key)
                        total_models = len(st.session_state['selected_models'])
                        for idx, model_id in enumerate(st.session_state['selected_models']):
                            ralf.load_and_process_data(
                                st.session_state['df'],
                                st.session_state['source_col'],
                                st.session_state['target_col'],
                                model_id
                            )
                            ralf.load_and_configure_model()
                            total_steps = epochs * (len(st.session_state['df']) // batch_size)
                            current_step = 0
                            def update_progress(step_info):
                                nonlocal current_step
                                current_step += 1
                                progress = min((idx + current_step / total_steps) / total_models, 1.0)
                                progress_bar.progress(progress)
                            ralf.trainer.add_callback(update_progress)
                            ralf.trainer.train()
                        progress_bar.progress(1.0)
                        st.success("Training completed for all recommended models!")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.info("Please analyze your dataset and get model recommendations first.")


if __name__ == "__main__":
    main()