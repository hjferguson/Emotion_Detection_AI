# Harlan Ferguson 101133838
#
# TensorFlow has proved to be difficult to work with when attempting to utilize my GPU.
# Due to frustration, I've moved over to PyTorch in hopes of getting my website to work in time for project submission.
#
# Attempt 1 with a basic sequential model resulted in 11% accuracy.
# Attempt 2, I used a similar model structure to my TensorFlow model. 53% accuracy.
# Tripled the epoch and got to 56% accuracy. 56% is not good enough. I'm hoping for closer to 70%.
#
# I've tried a few pretrained models so far. VGG16 underperformed, Resnet16 was close to my sequential model, resnet34 was around 59%
# I need to use a pretrained model no matter what... After some research, I'm going to try EfficientNet because it's known for
# efficiency and accuracy. Here's the updated code using EfficientNet-B0.
#
# EfficientNet resulted in a 68.2% accuracy!!!!!! so I'm going to use this for now, since this is the best result I've gotten so far.

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import os

# Creating a directory to save the generated graphs
graphs_dir = "graphs/efficientnet"
os.makedirs(graphs_dir, exist_ok=True)

BATCH_SIZE = 32


TRANSFORM = Compose([
    Resize((224, 224)),  
    Grayscale(num_output_channels=3), #efficientnet uses RGB images
    RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
    RandomHorizontalFlip(),
    RandomRotation(20),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

training_data = ImageFolder(root="../data/archive/train", transform=TRANSFORM)
test_data = ImageFolder(root="../data/archive/test", transform=TRANSFORM)

train_size = int(0.85 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Initialize and modify the pretrained EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)

history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    total_loss, total_correct = 0, 0
    for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1} Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_accuracy = total_correct / len(dataloader.dataset)
    print(f"Training Loss: {epoch_loss:>8f}, Accuracy: {(100*epoch_accuracy):>0.1f}%")
    history['train_loss'].append(epoch_loss)
    history['train_accuracy'].append(epoch_accuracy)

# Modified validate function
def validate(dataloader, model, loss_fn):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validating"):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_accuracy = total_correct / len(dataloader.dataset)
    print(f"Validation Loss: {epoch_loss:>8f}, Accuracy: {(100*epoch_accuracy):>0.1f}%")
    history['val_loss'].append(epoch_loss)
    history['val_accuracy'].append(epoch_accuracy)

# Prepare for training and validation
epochs = 11
for epoch in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer, epoch)
    validate(val_dataloader, model, loss_fn)
    scheduler.step(history['val_loss'][-1])

# Final evaluation on the test dataset
validate(test_dataloader, model, loss_fn)

# Saving the model
torch.save(model.state_dict(), "efficientnet_model.pth")
print("Training complete. Model saved as efficientnet_model.pth")

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.plot(history['train_accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{graphs_dir}/model_accuracy.png')

# Plot training & validation loss values
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{graphs_dir}/model_loss.png')

# Training Accuracy vs Loss
plt.figure(figsize=(10, 4))
plt.plot(history['train_accuracy'], label='Train Accuracy')
plt.plot(history['train_loss'], label='Train Loss', linestyle='--')
plt.title('Training Accuracy vs. Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(f'{graphs_dir}/training_accuracy_vs_loss.png')

# Validation Accuracy vs Loss
plt.figure(figsize=(10, 4))
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Validation Accuracy vs. Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(f'{graphs_dir}/validation_accuracy_vs_loss.png')