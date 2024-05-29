import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from Generator import Generator
from Discriminator import Discriminator
from Trainer import train 
from Data_Generator import synthetic_data_generator 
import json
import time

# Hyperparameter configuration
file_path = 'config.json'
with open(file_path, 'r') as file:
    hyperparameters = json.load(file)

batch_size = hyperparameters['batch_size']
latent_dim = hyperparameters['latent_dim']
gp_weight = hyperparameters['gp_weight']
discriminator_steps = hyperparameters['discriminator_steps']
epochs = hyperparameters['epochs']
feature_size = hyperparameters['feature_size']
fixed_range = hyperparameters['fixed_range']
fixed_features = hyperparameters['fixed_features']
output_dataset_size = hyperparameters['output_dataset_size']
feature_names = hyperparameters['feature_names']
num_samples = hyperparameters['num_samples']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load training data from CSV
csv_file = "./Training_data/training_data.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)
training_data = df.values.astype(np.float32)

# Prepare DataLoader
dataset = TensorDataset(torch.tensor(training_data))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Build models
generator = Generator(input_dim=latent_dim, output_dim=feature_size).to(device)
discriminator = Discriminator(input_dim=feature_size).to(device)

start_time = time.time()
# Train WGAN-GP
train(generator, discriminator, data_loader, batch_size, latent_dim, gp_weight, discriminator_steps, epochs, device, fixed_range, fixed_features, feature_size)
end_time = time.time()
execution_time = end_time - start_time
print("Training complete....")
print("Training time:", execution_time, "seconds")
print("Generating synthetic data....")

start_time = time.time()
# Generate and save synthetic data after training
synthetic_data_generator(generator, latent_dim, device, fixed_range, fixed_features, feature_names, output_dataset_size)
end_time = time.time()
execution_time = end_time - start_time
print("Synthetic Data Generation Complete....")
print("Synthetic Data Generation Time:", execution_time, "seconds")