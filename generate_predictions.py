import boto3
import pandas as pd
import json

# Initialize the boto3 client for SageMaker runtime
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Define the SageMaker endpoint name
endpoint_name = 'your-sagemaker-endpoint'

# Prepare your data for prediction
# Assuming you have your input data as a list of dictionaries
input_data = [
    {"feature1": value1, "feature2": value2},
    {"feature1": value3, "feature2": value4},
    # Add more data points as needed
]

# Convert the input data to JSON format if required
# This depends on how your model endpoint expects the input data
input_json = json.dumps(input_data)

# Define the content type and accept type
content_type = 'application/json'  # or 'text/csv' if your model accepts CSV
accept = 'application/json'  # or 'text/csv' if your model returns CSV

# Invoke the endpoint and get predictions
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Accept=accept,
    Body=input_json
)

# Parse the response
result = response['Body'].read().decode('utf-8')

# Assuming the response is JSON, parse it
predictions = json.loads(result)

# Convert predictions to a DataFrame
df_predictions = pd.DataFrame(predictions)

# Save predictions to a CSV file
df_predictions.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
