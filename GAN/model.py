import torch
import torch.nn as nn

# Generator Network (with LSTM)
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Generator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # Linear layer
        out = self.linear(out[:, -1, :])
        return out

# Discriminator Network (simple feed-forward network)
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Hyperparameters
G_INPUT_DIM = 100  # Random noise dimension coming into the Generator
G_HIDDEN_DIM = 128
G_OUTPUT_DIM = 1  # The dimension of the generated sequences by the Generator
D_INPUT_DIM = G_OUTPUT_DIM  # Discriminator input size
D_HIDDEN_DIM = 128

# Create the Generator and Discriminator
generator = Generator(G_INPUT_DIM, G_HIDDEN_DIM, G_OUTPUT_DIM)
discriminator = Discriminator(D_INPUT_DIM, D_HIDDEN_DIM)

print(generator)
print(discriminator)
