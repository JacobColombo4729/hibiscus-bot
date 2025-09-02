import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3

# --- 1. Configuration ---

# --- Essentials ---
role_arn = "arn:aws:iam::ACCOUNT_ID:role/YourSageMakerExecutionRole" # <-- IMPORTANT: Replace with your SageMaker Role ARN
s3_bucket = "your-sagemaker-fine-tuning-bucket" # <-- IMPORTANT: Replace with your S3 bucket name
s3_dataset_path = f"s3://{s3_bucket}/dataset"
s3_output_path = f"s3://{s3_bucket}/output"

# --- Training Job Details ---
job_name = "llama3-8b-classifier-finetune"
instance_type = "ml.g5.2xlarge" # A good starting point for 8B models
instance_count = 1
volume_size = 200 # in GB

# --- Model & Hyperparameters ---
# These hyperparameters should match or be passed to the training_script.py
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
epochs = 3
batch_size = 2

# You must have a 🤗 Hub account and accept the Llama 3 license to use it.
# Store your Hugging Face token in AWS Secrets Manager for security.
# Secret name should be something like 'huggingface-token' and the key inside 'HF_TOKEN'
aws_session = boto3.Session()
secrets_manager = aws_session.client("secretsmanager", region_name="us-east-1")
try:
    hf_token_secret = secrets_manager.get_secret_value(SecretId="huggingface-token")
    hf_token = hf_token_secret["SecretString"]
except Exception as e:
    print(f"ERROR: Could not retrieve Hugging Face token from Secrets Manager. Please ensure the secret exists.")
    print(f"Details: {e}")
    hf_token = None


# --- 2. Create the HuggingFace Estimator ---

# The estimator is the main object for configuring the SageMaker training job.
huggingface_estimator = HuggingFace(
    entry_point="training_script.py", # Your training script
    source_dir="./", # Directory containing the script
    instance_type=instance_type,
    instance_count=instance_count,
    volume_size=volume_size,
    role=role_arn,
    base_job_name=job_name,
    output_path=s3_output_path,
    
    # Specify the container image from SageMaker's Deep Learning Containers
    transformers_version="4.38",
    pytorch_version="2.2",
    py_version="py310",
    
    # Pass hyperparameters to the training script
    hyperparameters={
        "model_id": base_model_id,
        "epochs": epochs,
        "batch_size": batch_size,
    },
    
    # Pass the Hugging Face token securely
    environment={"HUGGING_FACE_HUB_TOKEN": hf_token} if hf_token else {},
)

# --- 3. Launch the Training Job ---

if hf_token:
    print(f"Starting SageMaker training job '{job_name}'...")
    print(f"Dataset will be read from: {s3_dataset_path}")
    print(f"Model artifacts will be saved to: {s3_output_path}")

    huggingface_estimator.fit({"train": s3_dataset_path}, wait=True)

    print("--- Training Job Complete ---")
    print(f"Model artifacts saved in: {huggingface_estimator.model_data}")
else:
    print("Could not start training job due to missing Hugging Face token.")
