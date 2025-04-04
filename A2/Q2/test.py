import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

# -----------------------
# 1. Dataset Loader
# -----------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, image_size=(28, 28)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_paths = []
        self.labels = []

        for label in sorted(os.listdir(root_dir)):  # Folders 0-9
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, img_file))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and preprocess image using Pillow
        image = Image.open(img_path).convert("L")  # Grayscale
        image = image.resize(self.image_size)       # Resize to 28x28

        # Convert image to numpy array and flatten
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image = (image - 0.5) / 0.5                        # Normalize to [-1, 1]
        image = image.flatten()                            # Flatten to (784,)
        
        # Convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_tensor, label_tensor
    
# -----------------------
# 2. Load the Dataset
# -----------------------
data_dir = "q2_data"
dataset_mlp = ImageDataset(data_dir)
dataloader_mlp = DataLoader(dataset_mlp, batch_size=64, shuffle=True)

# -----------------------
# 3. MLP Model
# -----------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes
        )

    def forward(self, x):
        return self.model(x)

# -----------------------
# 4. Training Loop
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader_mlp:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# -----------------------
# 5. Save the Model
# -----------------------
torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved successfully!")




# -----------------------
# 2. Load the Dataset
# -----------------------
data_dir = "q2_data"
dataset_cnn = ImageDataset(data_dir)
dataloader_cnn = DataLoader(dataset_cnn, batch_size=64, shuffle=True)

# -----------------------
# 3. CNN Model
# -----------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (28x28) → (28x28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # (28x28) → (14x14)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (14x14) → (14x14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                    # (14x14) → (7x7)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(128, 10)           # 10 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = self.fc_layers(x)
        return x

# -----------------------
# 4. Training Loop
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader_cnn:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# -----------------------
# 5. Save the Model
# -----------------------
torch.save(model.state_dict(), "cnn_model.pth")
print("CNN model saved successfully!")
