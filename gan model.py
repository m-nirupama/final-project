import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from resnet9 import ResNet9, Generator  # Import from your existing ResNet9-based GAN model

# ----------------------------
# Configuration
latent_dim = 100
image_channels = 3
image_size = 64
batch_size = 8
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model Initialization
generator = Generator(latent_dim, image_channels).to(device)
discriminator = ResNet9(in_channels=image_channels, num_diseases=1).to(device)

# ----------------------------
# Loss & Optimizers
adversarial_loss = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# ----------------------------
# Training Loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    for step in range(10):  # limited steps per epoch for illustration

        # Generate real and synthetic (generated) data
        real_images = torch.randn(batch_size, image_channels, image_size, image_size).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)

        noise = torch.randn(batch_size, latent_dim).to(device)
        generated_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # --------------------
        # Train Discriminator
        # --------------------
        optimizer_D.zero_grad()
        real_output = discriminator(real_images)
        fake_output = discriminator(generated_images.detach())

        d_loss_real = adversarial_loss(real_output, real_labels)
        d_loss_fake = adversarial_loss(fake_output, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # --------------------
        # Train Generator
        # --------------------
        optimizer_G.zero_grad()
        gen_output = discriminator(generated_images)
        g_loss = adversarial_loss(gen_output, real_labels)  # Want D to think fakes are real
        g_loss.backward()
        optimizer_G.step()

        if step % 2 == 0:
            print(f"Step {step} | Discriminator Loss: {d_loss.item():.4f} | Generator Loss: {g_loss.item():.4f}")

    # Save sample generated images
    save_image(generated_images[:4], f"generated_epoch{epoch+1}.png", normalize=True)
