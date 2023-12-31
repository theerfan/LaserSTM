import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from GAN.model import Generator, Discriminator

from Utilz.training import CustomSequence, load_model_params

import os

REAL_LABEL = 1
GEN_LABEL = 0


# Wild assumption: Latent dim is the same as the input dim
# Because we want to give the GAN our input data and have it generate the desired output!
def gan_train(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_epochs: int,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence = None,
    lr: float = 0.001,
    out_dir: str = ".",
    save_interval: int = None,
    batch_size: int = 200,
):
    # Assuming you're using a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # TODO: Turn these into input arguments
    # Create the Generator and Discriminator
    generator = Generator(input_dim, hidden_dim, output_dim)
    discriminator = Discriminator(input_dim, hidden_dim)

    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    else:
        pass

    # Create the loss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    generator_losses = []
    discriminator_losses = []
    if val_dataset is not None:
        val_losses = []

    for epoch in range(num_epochs):
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            ### (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # The logs are in the loss function!

            batch_size = X_batch.size(0)
            ## Train with real data
            discriminator.zero_grad()
            real_data = y_batch.to(device)
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
            generator_data = generator(X_batch)
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
            discriminator_losses.append(errD.item())
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
            generator_losses.append(errG.item())
            # Calculate gradients for G
            errG.backward()
            ## For printing: extract the scalar value of the loss function
            # D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Validation
            if val_dataset is not None:
                generator.eval()
                with torch.no_grad():
                    generator_data = generator(X_batch)
                    label.fill_(REAL_LABEL)  # fake labels are real for generator cost
                    # Since we just updated D,
                    # perform another forward pass of all-fake batch through D
                    output = discriminator(generator_data).view(-1)
                    # Calculate G's loss based on this output
                    errG = criterion(output, label)
                    val_losses.append(errG.item())
            else:
                pass

        if save_interval and epoch % save_interval == 0:
            # Save the models
            torch.save(generator.state_dict(), f"{out_dir}/generator.pt")
            torch.save(discriminator.state_dict(), f"{out_dir}/discriminator.pt")
        else:
            pass

        # Log the progress
        print(f"[{epoch}/{num_epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}")

    # Save the models
    torch.save(generator.state_dict(), "generator.pt")
    torch.save(discriminator.state_dict(), "discriminator.pt")

    return generator_losses, discriminator_losses, val_losses

