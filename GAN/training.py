import torch.optim as optim
import torch
import torch.nn as nn

from GAN.model import Generator, Discriminator

from train_utils.train_predict_utils import CustomSequence

REAL_LABEL = 1
GEN_LABEL = 0

# Assuming you're using a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Wild assumption: Latent dim is the same as the input dim
# Because we want to give the GAN our input data and have it generate the desired output!
def train_GAN(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_epochs: int,
    train_set: CustomSequence,
    batch_size: int,
    lr: float = 0.001,
):
    # Create the Generator and Discriminator
    generator = Generator(input_dim, hidden_dim, output_dim)
    discriminator = Discriminator(input_dim, hidden_dim)

    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Create the loss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    # TODO: Assuming you have a DataLoader `dataloader` that loads real sequences
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        for i, sample_generator in enumerate(train_set):
            ### (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # The logs are in the loss function!

            for X, y in sample_generator:
                ## Train with real data
                discriminator.zero_grad()
                real_data = y.to(device)
                batch_size = real_data.size(0)
                label = torch.full(
                    (batch_size,), REAL_LABEL, dtype=torch.float, device=device
                )
                # Forward pass real batch through D
                output = discriminator(real_data).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()

                ## Train with Generator-made data 
                # (would be `noise` in the original GAN idea)
                # Generate made-up data batch with G
                generator_data = generator(X)
                label.fill_(GEN_LABEL)
                # Classify all fake batch with D
                output = discriminator(generator_data.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_generator = criterion(output, label)
                # Calculate the gradients for this batch,
                # accumulated (summed) with previous gradients
                errD_generator.backward()
                ## For printing: extract the scalar value of the loss function
                # D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_generator
                # Update D
                optimizerD.step()

                ### (2) Update Generator: maximize log(D(G(z)))
                generator.zero_grad()
                label.fill_(REAL_LABEL)  # fake labels are real for generator cost
                # Since we just updated D,
                # perform another forward pass of all-fake batch through D
                output = discriminator(generator_data).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                ## For printing: extract the scalar value of the loss function
                # D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

        # Log the progress
        print(f"[{epoch}/{num_epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}")
