import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# The same network architecture as in training
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 1) Load the trained model
    model_path = "./models/cnn_trained_model.pth"
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2) Parse user input (a single number from command line argument)
    # For demonstration, we assume the user enters a pixel intensity (0-255)
    user_input = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

    # 3) Create a fake "image" by replicating the input value 784 times
    # Normalize by dividing by 255
    input_tensor = torch.full((1, 784), user_input, dtype=torch.float32) / 255.0

    # 4) Run the model and predict the class
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # 5) Print JSON for Node.js
    print(json.dumps({
        "input": user_input,
        "predicted_class": predicted_class
    }))
