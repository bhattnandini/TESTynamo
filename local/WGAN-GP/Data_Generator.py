import torch
import pandas as pd
import numpy as np

def synthetic_data_generator(generator, latent_dim, device, fixed_feature_values, fixed_feature_indices, feature_names, num_samples=1000, file_name="./Generated_data/synthetic_data.csv"):
    generator.eval()
    noise = torch.randn(num_samples, latent_dim).to(device)
    synthetic_data = generator(noise).cpu().detach().numpy()

    # # Modify the specific feature to pick a value from the fixed range
    # for index in range(6):
    #     fixed_values = torch.tensor(np.random.choice(fixed_feature_values, num_samples)).to(device)
    #     synthetic_data[:, fixed_feature_indices[index]] = fixed_values
    
    # Save to CSV
    df = pd.DataFrame(synthetic_data, columns=feature_names)
    df.to_csv(file_name, index=False)
    print(f"Synthetic data saved to {file_name}")