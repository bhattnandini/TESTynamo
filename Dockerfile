# Use an official PyTorch image as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /opt/ml/code

# Copy the training script into the container
COPY train.py requirements.txt training_data.csv /opt/ml/code/

# Install any additional dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Define the entrypoint to run the training script
ENTRYPOINT ["python", "train.py"]

# Provide default arguments
# CMD ["--latent_dim", "100", "--batch_size", "16", "--feature_size", "3", "--discriminator_steps", "5", "--epochs", "1000", "--gp_weight", "10.0"]