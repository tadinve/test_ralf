import os
import pandas as pd
import pickle
import warnings
# import torch
import psutil  # Add this import

import openai
from openai import OpenAI
import json
import re

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoConfig
from transformers.trainer_callback import TrainerCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset, ClassLabel, Features, Value

from transformers import AutoConfig

def estimate_param_count(model_id="distilbert-base-uncased"):
    try:
        config = AutoConfig.from_pretrained(model_id)

        # Get common configuration attributes, handling different names
        vocab_size = getattr(config, 'vocab_size', None)
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'dim', None))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', None))
        intermediate_size = getattr(config, 'intermediate_size', getattr(config, 'hidden_dim', None))
        num_attention_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_heads', None)) # Needed for some models

        if None in [vocab_size, hidden_size, num_layers, intermediate_size]:
             return "Error: Could not get all config attributes for parameter estimation."


        # This is a simplified estimation and may not be perfectly accurate for all models
        # It primarily covers BERT-like architectures

        # Embeddings
        embeddings = vocab_size * hidden_size

        # Transformer layers (simplified estimation covering common components)
        # This part might need further refinement for specific architectures
        # A more accurate way would involve iterating through model components, but this is more complex
        # Attempting a generalized approach based on common parameters

        # Attention parameters (simplified: QKV weights + output weights + biases)
        attn_params_per_head = hidden_size * (hidden_size // num_attention_heads) + (hidden_size // num_attention_heads) # QKV per head
        attn_output_per_layer = hidden_size * hidden_size + hidden_size # Output projection + bias
        total_attn_params_per_layer = num_attention_heads * attn_params_per_head * 3 + attn_output_per_layer # 3 for QKV

        # FFN parameters (input weight + output weight + biases)
        ffn_params_per_layer = hidden_size * intermediate_size + intermediate_size * hidden_size + intermediate_size + hidden_size # weights + biases

        # Layer Norm parameters (approximate: 2*hidden_size for gamma and beta)
        layer_norm_params_per_layer = 2 * hidden_size * 2 # Two layer norms per layer typically

        transformer_total = num_layers * (total_attn_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)


        total = embeddings + transformer_total # This estimation might still miss some parameters

        # Format the total parameter count with units M, G, B, or P
        if total >= 1e15:
            return f"{total / 1e15:.2f}P"
        elif total >= 1e12:
            return f"{total / 1e12:.2f}T" # Using T for trillion
        elif total >= 1e9:
            return f"{total / 1e9:.2f}B"
        elif total >= 1e6:
            return f"{total / 1e6:.2f}M"
        else:
            return str(total) # Return as string for smaller numbers

    except Exception as e:
        return f"Error estimating: {e}"


# Define the custom callback for saving the Ralf instance
class RalfSavingCallback(TrainerCallback):
    """
    A custom callback to save the Ralf instance periodically during training.
    """
    def __init__(self, ralf_instance, save_path="ralf_state.pkl"):
        self.ralf_instance = ralf_instance
        self.save_path = save_path

    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint is saved.
        """
        print(f"Saving Ralf state at step {state.global_step}...")
        self.ralf_instance.save_state(file_path=self.save_path)
        print("Ralf state saved.")

# Define the Ralf class
class Ralf:
    """
    A class to encapsulate the datasets, model, and trainer for the Ralf project.
    """
    def __init__(self, HF_TOKEN, OPENAI_API_KEY=None, GEMINI_API_KEY=None): # Made HF_TOKEN required and others optional for recommendation
        """
        Initializes the Ralf class with placeholders for datasets, model name, and trainer.
        Requires HF_TOKEN and at least one of OPENAI_API_KEY or GEMINI_API_KEY for recommendations.
        """

        # Validate required keys
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is required.")
        if not OPENAI_API_KEY and not GEMINI_API_KEY:
            raise ValueError("Either OPENAI_API_KEY or GEMINI_API_KEY must be provided for recommendations.")


        # Hardware checks
        self.gpu_available = None
        self.gpu_count = None
        self.gpu_name = None
        self.gpu_ram_gb = None
        self.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        """
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else None
        self.gpu_ram_gb = None
        if self.gpu_available:
            props = torch.cuda.get_device_properties(0)
            self.gpu_ram_gb = round(props.total_memory / (1024 ** 3), 2)
        self.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        """

        print(f"GPU available: {self.gpu_available}")
        if self.gpu_available:
            print(f"GPU count: {self.gpu_count}")
            print(f"GPU name: {self.gpu_name}")
            print(f"GPU RAM: {self.gpu_ram_gb} GB")
        print(f"Available system RAM: {self.ram_gb} GB")

        self.golden_dataset = None
        self.platinum_dataset = None
        # Add other datasets as needed
        self.other_datasets = {}
        self.model_name = None
        self.trainer = None
        self.num_labels = None
        self.label_to_id = None
        self.id_to_label = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.model = None

        # API keys
        self.open_api_key = OPENAI_API_KEY
        self.gemini_key = GEMINI_API_KEY
        self.hf_token = HF_TOKEN # Stored the HF_TOKEN

    def set_keys(self, open_api_key=None, gemini_key=None, hf_token=None):
        """
        Set API keys for OpenAI, Gemini, and Hugging Face.
        Validates that at least one of open_api_key or gemini_key is provided if setting either.
        """
        if hf_token is not None:
            self.hf_token = hf_token

        if open_api_key is not None or gemini_key is not None:
             if open_api_key is None and gemini_key is None:
                 raise ValueError("Either open_api_key or gemini_key must be provided.")
             self.open_api_key = open_api_key
             self.gemini_key = gemini_key


    def load_and_process_data(self, df: pd.DataFrame, text_column: str, label_column: str, model_name: str):
        """
        Loads, processes, and tokenizes the data, and splits it into training and validation sets.

        Args:
            df: The input pandas DataFrame.
            text_column: The name of the column containing the text data.
            label_column: The name of the column containing the labels.
            model_name: The name of the pre-trained model to load (e.g., "bert-base-uncased").
        """
        self.model_name = model_name # Set model_name here

        # Ensure the DataFrame has 'text' and 'label' columns
        if text_column != 'text' or label_column != 'label':
            df = df.rename(columns={text_column: 'text', label_column: 'label'})

        # Determine unique labels and create mappings
        unique_conditions = df['label'].unique().tolist()
        self.num_labels = len(unique_conditions)
        self.label_to_id = {condition: i for i, condition in enumerate(unique_conditions)}
        self.id_to_label = {i: condition for i, condition in enumerate(unique_conditions)}

        # Map string labels to integer IDs
        df['label'] = df['label'].map(self.label_to_id)

        # Select only the 'text' and 'label' columns for the dataset
        dataset_df = df[['text', 'label']]

        # Convert pandas DataFrame to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(dataset_df)

        # Convert the 'label' column to ClassLabel
        features = hf_dataset.features.copy()
        features['label'] = ClassLabel(num_classes=self.num_labels, names=unique_conditions)
        hf_dataset = hf_dataset.cast(features)


        # Split the dataset into training and validation sets
        train_df, val_df = train_test_split(
            dataset_df,
            test_size=0.2,
            random_state=42, # for reproducibility
            stratify=dataset_df['label'] # Stratify to maintain class distribution
        )

        # Convert split DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)


        # Initialize the tokenizer using the model name
        if self.model_name is None:
            raise ValueError("model_name must be set before calling load_and_process_data")
        # Use HF_TOKEN if available when loading the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        # Tokenize the training and validation datasets
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Remove unnecessary columns (original text column and any extra index columns)
        self.train_dataset = self.train_dataset.remove_columns(['text', '__index_level_0__']) # '__index_level_0__' is added by from_pandas
        self.val_dataset = self.val_dataset.remove_columns(['text', '__index_level_0__'])

        print("Data loading and processing completed.")
        print(f"Number of labels: {self.num_labels}")
        print("Label mapping:", self.label_to_id)

    def load_and_configure_model(self): # Removed model_name argument
    # When we start to train / fine-tune models, we will most-likely send the config and data to
    # another container running the training process. If it is in the cloud, it can be turn up and shut down
    # as needed to reduce the cost of ownership.
        ''' """
        Loads a pre-trained model and configures it for sequence classification with LoRA.

        Args:
            model_name: The name of the pre-trained model to load (e.g., "bert-base-uncased").
        """
        # Use self.model_name
        # Use HF_TOKEN if available when loading the model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels, token=self.hf_token)

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,  # LoRA attention dimension
            lora_alpha=16,  # Alpha parameter for LoRA scaling
            target_modules=["query", "value"],  # Modules to apply LoRA to
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            bias="none",  # Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
            task_type="SEQ_CLS",  # Task type, e.g. "SEQ_CLS" for sequence classification
        )

        # Apply the LoRA configuration
        self.model = get_peft_model(self.model, lora_config)

        # Print the trainable parameters
        self.model.print_trainable_parameters()

        print(f"Model loading and LoRA setup completed for '{self.model_name}'.")
        '''

    def initialize_trainer(self, output_dir: str = "./results", save_path: str = "ralf_state.pkl"):
        """
        Intializes the Hugging Face Trainer object for training.

        Args:
            output_dir: The output directory for model checkpoints and logs.
            save_path: The path to save the Ralf state using the custom callback.
        """
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,  # Output directory for model checkpoints and logs
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=16,  # Batch size for training
            per_device_eval_batch_size=16,  # Batch size for evaluation
            warmup_steps=500,  # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # Strength of weight decay
            logging_dir="./logs",  # Directory for storing logs
            logging_steps=10, # Log every 10 steps
            eval_strategy="epoch", # Evaluate at the end of each epoch
            save_strategy="epoch", # Save checkpoint at the end of each epoch
            load_best_model_at_end=True, # Load the best model at the end of training
            metric_for_best_model="eval_loss", # Metric to use for loading the best model
            greater_is_better=False, # For eval_loss, lower is better
            report_to="none", # Disable reporting to services like W&B for simplicity
            hub_token=self.hf_token, # Pass HF token to trainer for potentially pushing to hub
            hub_model_id=f"my-awesome-model-{os.path.basename(output_dir)}" # Example hub_model_id
        )

        # Initialize the custom callback
        ralf_saving_callback = RalfSavingCallback(self, save_path=save_path)

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,  # Our LoRA-configured model
            args=training_args,  # Training arguments
            train_dataset=self.train_dataset,  # Training dataset
            eval_dataset=self.val_dataset,  # Validation dataset
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer), # Use DataCollatorWithPadding
            # compute_metrics=compute_metrics # Optional: define a function to compute metrics
            callbacks=[ralf_saving_callback] # Add the custom callback here
        )

        print("Trainer initialization completed with RalfSavingCallback.")

    def save_state(self, file_path: str = "ralf_state.pkl"):
        """
        Saves the current state of the Ralf instance using pickling.

        Args:
            file_path: The path to the file where the state will be saved.
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Ralf state successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving Ralf state: {e}")

    @staticmethod
    def load_state(file_path: str = "ralf_state.pkl"):
        """
        Loads a previously saved Ralf instance from a pickle file.

        Args:
            file_path: The path to the pickle file.

        Returns:
            The loaded Ralf instance, or None if loading fails.
        """
        try:
            with open(file_path, 'rb') as f:
                ralf_instance = pickle.load(f)
            print(f"Ralf state successfully loaded from {file_path}")
            return ralf_instance
        except FileNotFoundError:
            print(f"Error loading Ralf state: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading Ralf state: {e}")
            return None

    def format_param_size(self, total_params):
        """Formats parameter count with units M, B, or P."""
        if total_params >= 1e15:
            return f"{total_params / 1e15:.2f}P"
        elif total_params >= 1e12:
            return f"{total_params / 1e12:.2f}T" # Using T for trillion
        elif total_params >= 1e9:
            return f"{total_params / 1e9:.2f}B"
        elif total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        else:
            return str(total_params) # Return as string for smaller numbers


    def estimate_param_count(self, model_id="distilbert-base-uncased"):
        """Estimates parameter count for a given model ID."""
        try:
            # Pass HF_TOKEN if available when loading the config
            config = AutoConfig.from_pretrained(model_id, token=self.hf_token)

            # Get common configuration attributes, handling different names
            vocab_size = getattr(config, 'vocab_size', None)
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'dim', None))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', None))
            intermediate_size = getattr(config, 'intermediate_size', getattr(config, 'hidden_dim', None))
            num_attention_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_heads', None))

            if None in [vocab_size, hidden_size, num_layers, intermediate_size, num_attention_heads]:
                 return "Error: Could not get all config attributes for parameter estimation."

            # This is a simplified estimation and may not be perfectly accurate for all models
            # It primarily covers BERT-like architectures

            # Embeddings
            embeddings = vocab_size * hidden_size

            # Transformer layers (simplified estimation covering common components)
            # Attention parameters (simplified: QKV weights + output weights + biases)
            attn_params_per_head = hidden_size * (hidden_size // num_attention_heads) + (hidden_size // num_attention_heads) # QKV per head
            attn_output_per_layer = hidden_size * hidden_size + hidden_size # Output projection + bias
            total_attn_params_per_layer = num_attention_heads * attn_params_per_head * 3 + attn_output_per_layer # 3 for QKV

            # FFN parameters (input weight + output weight + biases)
            ffn_params_per_layer = hidden_size * intermediate_size + intermediate_size * hidden_size + intermediate_size + hidden_size # weights + biases

            # Layer Norm parameters (approximate: 2*hidden_size for gamma and beta)
            layer_norm_params_per_layer = 2 * hidden_size * 2 # Two layer norms per layer typically

            transformer_total = num_layers * (total_attn_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)

            total = embeddings + transformer_total # This estimation might still miss some parameters

            return self.format_param_size(total)

        except Exception as e:
            return f"Error estimating: {e}"


    def recommend(self, input_csv_file,
                source_col,
                target_col,
                analysis,
                model_selected = "gpt-4o-mini"):
        """Recommends top 3 LLMs for fine-tuning and a golden dataset based on problem type and resources using GPT-4o-mini or Gemini."""

        # gpu_available = self.gpu_available
        # gpu_ram_gb = self.gpu_ram_gb
        # ram_gb = self.ram_gb
        gpu_available = 16
        gpu_ram_gb = 128
        ram_gb = 256

        if self.open_api_key:
            client = OpenAI(api_key=self.open_api_key)
            model_to_use = model_selected # Or another suitable OpenAI model
            print("Using OpenAI API for recommendations.")
        elif self.gemini_key:
             # Need to implement Gemini API calls if Gemini key is provided
             # This is a placeholder - actual Gemini implementation would go here
             # import google.generativeai as genai
             # genai.configure(api_key=self.gemini_key)
             # client = genai.GenerativeModel('gemini-pro') # Or another suitable Gemini model
             # model_to_use = 'gemini-pro'
             # print("Using Gemini API for recommendations.")
             # For now, if only Gemini key is provided, return an error or use a default
             llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters"])
             dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
             return llm_df, dataset_df, "Error: Gemini API implementation is not yet available in Ralf."

        else:
            # This case should ideally not be reached due to the __init__ validation
            llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters"])
            dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
            return llm_df, dataset_df, "Error: Neither OpenAI nor Gemini API key provided (should have been caught in initialization)."

        problem_types = analysis.get('types', [])
        reasoning = analysis.get('reasoning', 'No reasoning provided.')

        # Prompt for LLM recommendations and Hugging Face links
        llm_recommendation_prompt = (
            f"Based on the following problem types ({', '.join(problem_types)}) "
            f"and the available resources (GPU: {gpu_available}, GPU RAM: {gpu_ram_gb} GB, System RAM: {ram_gb} GB), "
            "recommend the top 5 open-source LLM models that would be suitable for fine-tuning for this task. "
            "For each recommended model, provide its name, the URL of its Hugging Face page, and a common identifier or path used to load the model (e.g., 'bert-base-uncased'). "
            "Return a JSON object with a single key 'llm_info' which is a list of dictionaries, where each dictionary contains 'name', 'huggingface_url', and 'model_id'. "
            "Do not include any other text, just the JSON object.\n\n"
            f"Problem types reasoning: {reasoning}"
        )

        # Prompt for Golden Dataset recommendation
        dataset_recommendation_prompt = (
            f"Based on the problem types ({', '.join(problem_types)}) and the sample data provided earlier (related to drug reviews and conditions), "
            "recommend a suitable publicly available golden dataset for fine-tuning LLMs for this task. "
            "The dataset should be available on platforms like Hugging Face Datasets or Kaggle. "
            "Provide the name of the dataset, a link to its source (Hugging Face or Kaggle), and identify the column names that would typically serve as the 'source' and 'target' columns for fine-tuning. "
            "Return a JSON object with a single key 'golden_dataset_info' which is a dictionary containing 'name', 'url', 'source_column', and 'target_column'. "
            "Do not include any other text, just the JSON object.\n\n"
            f"Problem types reasoning: {reasoning}"
        )


        try:
            # Get LLM recommendations and links using the selected client and model
            llm_response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": llm_recommendation_prompt}],
                max_tokens=512,
                temperature=0.1,
            )
            llm_content = llm_response.choices[0].message.content
            llm_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
            llm_recommendations = json.loads(llm_match.group(0)) if llm_match else json.loads(llm_content)


            if not isinstance(llm_recommendations, dict) or 'llm_info' not in llm_recommendations or not isinstance(llm_recommendations['llm_info'], list):
                 llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters"])
                 dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
                 return llm_df, dataset_df, analysis # Return empty DataFrames and the analysis


            recommended_llm_info = llm_recommendations['llm_info']


            # Get Golden Dataset recommendation using the selected client and model
            dataset_response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": dataset_recommendation_prompt}],
                max_tokens=256,
                temperature=0.1,
            )
            dataset_content = dataset_response.choices[0].message.content
            dataset_match = re.search(r'\{.*\}', dataset_content, re.DOTALL)
            golden_dataset_info = json.loads(dataset_match.group(0)) if dataset_match else json.loads(dataset_content)


            if not isinstance(golden_dataset_info, dict) or 'golden_dataset_info' not in golden_dataset_info or not isinstance(golden_dataset_info['golden_dataset_info'], dict):
                 llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters"])
                 dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
                 return llm_df, dataset_df, analysis # Return empty DataFrames and the analysis


            golden_dataset_details = golden_dataset_info['golden_dataset_info']


            # --- Create and populate the LLM DataFrame ---
            llm_data = []
            # Use the estimate_param_count method for parameter size
            for llm_info in recommended_llm_info:
                model_id = llm_info.get("model_id", "N/A")
                param_size = "N/A" # Default value
                if model_id != "N/A":
                    try:
                        # Use the instance method estimate_param_count
                        param_size = self.estimate_param_count(model_id)
                    except Exception as e:
                        param_size = f"Error estimating: {e}" # Indicate if there was an error

                llm_data.append({
                    "Name": llm_info.get("name", "N/A"),
                    "Hugging Face URL": llm_info.get("huggingface_url", "N/A"),
                    "Model ID": model_id,
                    "Parameters": param_size # Add the parameter size
                })

            llm_df = pd.DataFrame(llm_data)

            # --- Create and populate the Dataset DataFrame ---
            dataset_data = [{
                "Name": golden_dataset_details.get("name", "N/A"),
                "URL": golden_dataset_details.get("url", "N/A"),
                "Source Column": golden_dataset_details.get("source_column", "N/A"),
                "Target Column": golden_dataset_details.get("target_column", "N/A")
            }]
            dataset_df = pd.DataFrame(dataset_data)

            return llm_df, dataset_df, analysis # Return the DataFrames and analysis


        except Exception as e:
           # Return empty DataFrames and the error message if API call fails
           llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters"])
           dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
           return llm_df, dataset_df, f"Error calling API: {e}"


    def analyze_problem_type(self, df, source_col, target_col, model_selected = "gpt-4o-mini"):
        """Analyze the problem type using GPT-4o-mini based on selected columns."""
        print("Entering Problem Type Analysis")
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
            "Only choose from this list. Return a JSON object with two keys: 'types' (a list of the chosen types) and 'reasoning' (a string explaining your reasoning for each category of the selection type, for example, why is it classification or why it is reason). "
            "Do not include any other text, just the JSON object.\n\n"
            f"Source column: {source_col}\n"
            f"Target column: {target_col}\n"
            f"Sample data:\n{sample_text}"
        )
        # Use the appropriate client and model based on available keys
        if self.open_api_key:
            client = OpenAI(api_key=self.open_api_key)
            model_to_use = model_selected
        elif self.gemini_key:
            # Placeholder for Gemini implementation
            # import google.generativeai as genai
            # genai.configure(api_key=self.gemini_key)
            # client = genai.GenerativeModel('gemini-pro')
            # model_to_use = 'gemini-pro'
            return "Error: Gemini API implementation is not yet available for analysis."
        else:
            return "Error: No API key provided for analysis."


        print("Sending data to LLM for analysis")
        try:
            response = client.chat.completions.create(
                model=model_to_use,
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
            return f"Error calling API for analysis: {e}"