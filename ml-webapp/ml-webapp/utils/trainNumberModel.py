import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json

# Updated network for 784 input features (28x28 images) and 10 classes
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # from 784 input features to 128 neurons
        self.fc2 = nn.Linear(128, 64)   # hidden layer
        self.fc3 = nn.Linear(64, 10)    # output layer for 10 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 1) Load the dataset (adjust the file path if needed)
    file_path = "./data/cnn_dataset.csv"
    df = pd.read_csv(file_path)

    # 2) Separate features and labels
    #    - Features: all columns except the last ("label")
    #    - Labels: the "label" column
    X = df.iloc[:, :-1].values  # shape: (num_samples, 784)
    y = df.iloc[:, -1].values   # shape: (num_samples,)

    # Normalize pixel values (assuming they range 0-255)
    X = X / 255.0

    # Convert to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.long)

    # 3) Initialize model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4) Train loop
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 5) Save the trained model
    model_path = "./models/cnn_trained_model.pth"
    torch.save(model.state_dict(), model_path)

    # 6) Print JSON for Node.js to parse
    print(json.dumps({
        "message": "Model trained successfully with CNN dataset!",
        "model_path": model_path,
        "final_loss": loss.item(),
        "epochs": epochs
    }))
