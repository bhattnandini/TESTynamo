import sagemaker

# Load the model from the output path
model_data = 'S3 bucket location'

# Define IAM role
role = 'SageMaker IAM role'

# Initialize the model
model = sagemaker.model.Model(
    model_data=model_data,
    role=role,  
    image_uri='Container Image URI'  # Replace with your Docker image URI
)

# Deploy the model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
