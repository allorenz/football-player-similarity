import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)  # Bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Output layer (reconstruction)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_embeddings(self, data_tensor):
        # Extract embeddings from the encoder
        with torch.no_grad():
            embeddings = self.encoder(data_tensor)
        return embeddings.cpu().numpy() 
    

def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    """
    Train a given model with specified parameters.

    Args:
        model: The model to train.
        dataloader: The data loader providing the input data.
        criterion: The loss function (e.g., reconstruction loss).
        optimizer: The optimizer (e.g., Adam, SGD).
        num_epochs: Number of epochs to train the model (default: 50).
        log_interval: Interval for printing the loss during training (default: 10 epochs).

    Returns:
        Trained model.
    """
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data  # Inputs are the data
            optimizer.zero_grad()

            # Forward pass: encode-decode
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Reconstruction loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Logging every log_interval epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model