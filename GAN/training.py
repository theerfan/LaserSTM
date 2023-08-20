import torch.optim as optim
import torch
import torch.nn as nn

from GAN.model import Generator, Discriminator

# Hyperparameters
BATCH_SIZE = 64
LR = 0.0002
NUM_EPOCHS = 1000
LATENT_DIM = 100
REAL_LABEL = 1
FAKE_LABEL = 0

# Assuming you're using a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
G_INPUT_DIM = 100  # Random noise dimension coming into the Generator
G_HIDDEN_DIM = 128
G_OUTPUT_DIM = 1  # The dimension of the generated sequences by the Generator
D_INPUT_DIM = G_OUTPUT_DIM  # Discriminator input size
D_HIDDEN_DIM = 128

# Create the Generator and Discriminator
generator = Generator(G_INPUT_DIM, G_HIDDEN_DIM, G_OUTPUT_DIM)
discriminator = Discriminator(D_INPUT_DIM, D_HIDDEN_DIM)

# Move models to device
generator = generator.to(device)
discriminator = discriminator.to(device)

# Create the loss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

# Placeholder for input noise to Generator
noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(device)

# TODO: Assuming you have a DataLoader `dataloader` that loads real sequences
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader, 0):
        # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        
        # Train with real data
        discriminator.zero_grad()
        real_data = data.to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=device)
        
        output = discriminator(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        # Train with fake data
        noise.data.normal_()
        fake_data = generator(noise)
        label.fill_(FAKE_LABEL)
        output = discriminator(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update Generator: maximize log(D(G(z)))
        generator.zero_grad()
        label.fill_(REAL_LABEL)
        output = discriminator(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

    # Log the progress
    print(f"[{epoch}/{NUM_EPOCHS}] Loss_D: {errD.item()} Loss_G: {errG.item()}")
