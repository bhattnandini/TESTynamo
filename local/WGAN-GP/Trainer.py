import torch
import torch.optim as optim
import numpy as np
from Loss_Function import gradient_penalty

def train_step(generator, discriminator, real_data, batch_size, latent_dim, gp_weight, discriminator_steps, device, optimizer_G, optimizer_D, fixed_feature_values, fixed_feature_indices, feature_size):
    generator.train()
    discriminator.train()

    for _ in range(discriminator_steps):
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise)

        # # Modify the specific feature to pick a value from the fixed range
        # for index in range(feature_size - 1):
        #     fixed_values = torch.tensor(np.random.choice(fixed_feature_values, batch_size)).to(device)
        #     fake_data[:, fixed_feature_indices[index]] = fixed_values

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


def train(generator, discriminator, data_loader, batch_size, latent_dim, gp_weight, discriminator_steps, epochs, device, fixed_features, fixed_feature_indices, feature_size):
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for real_data in data_loader:
            real_data = real_data[0]
            if real_data.size(0) != batch_size:
                continue  # Skip the last batch if it's not the full batch size

            discriminator_loss, generator_loss = train_step(generator, discriminator, real_data, batch_size, latent_dim, gp_weight, discriminator_steps, device, optimizer_G, optimizer_D, fixed_features, fixed_feature_indices, feature_size)

        # if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}')