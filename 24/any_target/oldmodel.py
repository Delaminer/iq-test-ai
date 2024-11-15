mport torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Outputs should be between 0 and 1
        return x
# Set the random seed for reproducibility
torch.manual_seed(42)
# Define the parameters
input_dim = 4
hidden_dim = 16
output_dim = 4
learning_rate = 0.01
num_epochs = 100
batch_size = 16
# Generate some dummy data for training
num_samples = 1000
X = np.random.rand(num_samples, input_dim).astype(np.float32)  # Random input data
y = np.random.randint(0, 2, (num_samples, output_dim)).astype(np.float32)  # Random binary targets
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
# Create DataLoader for batching
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Initialize the model, loss function, and optimizer
model = SimpleMLP(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-attribute classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, targets)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Print the average loss for this epoch
    avg_loss = total_loss / len(dataloader)
    print(f”Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}“)
print(“Training complete!“)
# Test the model on a new sample
model.eval()
test_input = torch.tensor([[0.1, 0.5, 0.3, 0.9]], dtype=torch.float32)
prediction = model(test_input)
print(“Test Input:“, test_input)
print(“Prediction:“, prediction)