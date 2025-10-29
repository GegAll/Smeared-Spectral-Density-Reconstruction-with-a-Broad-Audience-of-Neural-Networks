import torch
import torch.nn as nn

# --- Model Configuration ---
TIME_EXTENT = 32
OUTPUT_FEATURES = 47
LEAKY_RELU_SLOPE = 0.2

class ArcS_t32(nn.Module):
    """
    Adapted 'Small' architecture for TIME_EXTENT=32.
    - 2 Conv1D layers (16, 32 maps)
    - 1 Fully Connected hidden layer (256 neurons)
    """
    def __init__(self):
        super(ArcS_t32, self).__init__()
        
        # Convolutional part
        self.convolutional_layers = nn.Sequential(
            # Input shape: (batch_size, 1, 32)
            # padding=1, kernel=3, stride=2 results in halving the sequence length
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            # Shape after Conv1: (batch_size, 16, 16)
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
            # Shape after Conv2: (batch_size, 32, 8)
        )
        
        self.flatten = nn.Flatten()
        # Shape after Flatten: (batch_size, 32 * 8 = 256)
        
        # Fully connected part
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=32 * 8, out_features=256),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Linear(in_features=256, out_features=OUTPUT_FEATURES) # Output layer
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fully_connected_layers(x)
        return x

class ArcM_t32(nn.Module):
    """
    Adapted 'Medium' architecture for TIME_EXTENT=32.
    - 2 Conv1D layers (32, 64 maps)
    - 1 Fully Connected hidden layer (256 neurons)
    """
    def __init__(self):
        super(ArcM_t32, self).__init__()
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            # Shape after Conv1: (batch_size, 32, 16)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
            # Shape after Conv2: (batch_size, 64, 8)
        )
        
        self.flatten = nn.Flatten()
        # Shape after Flatten: (batch_size, 64 * 8 = 512)
        
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=64 * 8, out_features=256),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Linear(in_features=256, out_features=OUTPUT_FEATURES)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fully_connected_layers(x)
        return x

class ArcL_t32(nn.Module):
    """
    Adapted 'Large' architecture for TIME_EXTENT=32.
    - 2 Conv1D layers (64, 128 maps)
    - 1 Fully Connected hidden layer (256 neurons)
    """
    def __init__(self):
        super(ArcL_t32, self).__init__()
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            # Shape after Conv1: (batch_size, 64, 16)
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
            # Shape after Conv2: (batch_size, 128, 8)
        )
        
        self.flatten = nn.Flatten()
        # Shape after Flatten: (batch_size, 128 * 8 = 1024)
        
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=128 * 8, out_features=256),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Linear(in_features=256, out_features=OUTPUT_FEATURES)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fully_connected_layers(x)
        return x