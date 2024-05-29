import torch
import torch.nn as nn
import torch.optim as optim
# import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sagemaker.s3 import S3Downloader

class Discriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand(real_data.size())

    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    prob_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()

def train_step(generator, discriminator, real_data, batch_size, latent_dim, gp_weight, discriminator_steps, device, optimizer_G, optimizer_D):
    generator.train()
    discriminator.train()

    for _ in range(discriminator_steps):
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise)

        real_data = real_data.to(device)
        fake_data = fake_data.to(device)

        optimizer_D.zero_grad()
        real_validity = discriminator(real_data)
        fake_validity = discriminator(fake_data)
        gp = gradient_penalty(discriminator, real_data, fake_data, device)
        discriminator_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp_weight * gp

        discriminator_loss.backward()
        optimizer_D.step()

    optimizer_G.zero_grad()
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_data = generator(noise)
    fake_validity = discriminator(fake_data)
    generator_loss = -torch.mean(fake_validity)

    generator_loss.backward()
    optimizer_G.step()

    return discriminator_loss.item(), generator_loss.item()


def train(generator, discriminator, data_loader, batch_size, latent_dim, gp_weight, discriminator_steps, epochs, device, feature_size):
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for real_data in data_loader:
            real_data = real_data[0]
            if real_data.size(0) != batch_size:
                continue  # Skip the last batch if it's not the full batch size

            discriminator_loss, generator_loss = train_step(generator, discriminator, real_data, batch_size, latent_dim, gp_weight, discriminator_steps, device, optimizer_G, optimizer_D)

        print(f'Epoch: {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}')

if __name__ == '__main__':
    print("enter main")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("GPU: ",torch.cuda.is_available())
    # print("before parsing args")
    # parser = argparse.ArgumentParser()
    # print("after parser")
    # parser.add_argument('--train', type=str, default=os.environ)
    # parser.add_argument('--latent_dim', type=int, default=100)
    # parser.add_argument('--feature_size', type=int, default=3)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--discriminator_steps', type=int, default=5)
    # parser.add_argument('--epochs', type=int, default=1000)
    # parser.add_argument('--gp_weight', type=int, default=10.0)
    # parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
   
    # print("after adding arguments")
    # args = parser.parse_args()
    # print("latent dim: ", args.latent_dim)
    # print("after parsing args")
    
    latent_dim = 100
    feature_size = 3
    batch_size = 16
    discriminator_steps = 5
    epochs = 10
    gp_weight = 10.0

    # Set AWS credentials and region

    # s3_path = 'training dataset location'

    # Define the local path where you want to save the dataset
    local_path = '/opt/ml/code/training_data.csv'

    # Download the file from S3
    # S3Downloader.download(s3_path, local_path)
    # print("after downloading csv")
    # Load training data from CSV
    df = pd.read_csv(local_path)
    print("after df")
    training_data = df.values.astype(np.float32)
    print("after training_data")
    # Prepare DataLoader
    dataset = TensorDataset(torch.tensor(training_data))
    print("after dataset")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("after data_loader")

    # Build models
    generator = Generator(input_dim=latent_dim, output_dim=feature_size).to(device)
    print("after generator")
    discriminator = Discriminator(input_dim=feature_size).to(device)
    print("after discriminator")

    train(generator, discriminator, data_loader, batch_size, latent_dim, gp_weight, discriminator_steps, epochs, device, feature_size)