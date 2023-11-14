import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

from GAN.model import Generator, Discriminator

from Utilz.training import CustomSequence

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
    train_set: CustomSequence,
    val_set: CustomSequence = None,
    lr: float = 0.001,
    out_dir: str = ".",
    save_interval: int = None,
):
    # Assuming you're using a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    if val_set is not None:
        val_losses = []

    for epoch in range(num_epochs):
        for i, sample_generator in enumerate(train_set):
            ### (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # The logs are in the loss function!

            for X, y in sample_generator:
                batch_size = X.size(0)
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
            if val_set is not None:
                generator.eval()
                with torch.no_grad():
                    generator_data = generator(X)
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


# Here we switch it up and only care about
# the MSE loss between the generated and the real data
def gan_predict(
    generator: nn.Module,
    model_param_path: str = None,
    test_dataset: CustomSequence = None,
    output_dir: str = ".",
    output_name: str = "all_preds.npy",
    verbose: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model parameters if path is provided
    if model_param_path is not None:
        torch.cuda.empty_cache
        params = torch.load(model_param_path, map_location=device)
        # remove the 'module.' prefix from the keys
        params = {k.replace("module.", ""): v for k, v in params.items()}
        generator.load_state_dict(params, strict=False)
    else:
        pass

    # Check if the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
    else:
        print("Warning: Data parallelism not available, using single GPU instead.")

    generator = generator.to(device)
    generator.eval()

    all_preds = []
    final_shape = None

    if verbose:
        print("Finished loading the model, starting prediction.")

    dataset_len = len(test_dataset)

    with torch.no_grad():
        for j in range(dataset_len):
            sample_generator = test_dataset[j]

            if verbose:
                print(
                    f"Processing batch {(j+1)} / {len(test_dataset)}"
                )

            counter = 0
            for X, y in sample_generator:
                if verbose:
                    print(f"Processing sample {(counter+1)} / n")
                    counter += 1

                X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)

                if final_shape is None:
                    final_shape = X.shape[-1]

                for _ in range(100):
                    pred = generator(X)
                    X = X[:, 1:, :]  # pop first

                    # add to last
                    X = torch.cat((X, torch.reshape(pred, (-1, 1, final_shape))), 1)

                all_preds.append(pred.squeeze())

            if verbose:
                print(f"Finished processing samples in {j} batch.")

        if verbose:
            print("Finished processing all batches.")

    all_preds = torch.stack(all_preds, dim=0).cpu().numpy()
    np.save(os.path.join(output_dir, f"{output_name}"), all_preds)
    return all_preds
