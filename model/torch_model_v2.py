# Harlan Ferguson 101133838
#
# TensorFlow has proved to be difficult to work with when attempting to utilize my GPU.
# Due to frustration, I've moved over to PyTorch in hopes of getting my website to work in time for project submission.
#
# Attempt 1 with a basic sequential model resulted in 11% accuracy.
# Attempt 2, I used a similar model structure to my TensorFlow model. 53% accuracy.
# Tripled the epoch and got to 56% accuracy. 56% is not good enough. I'm hoping for closer to 70%.
#
#I've tried a few pretrained models so far. VGG16 underpreformed, Resnet16 was close to my sequential model, resnet34 was around 59%
# I need to use a pretrained model no matter what... After some research, I'm going to try EfficientNet because it's known for
# efficiency and accuracy. Here's the updated code using EfficientNet-B0.
#
#efficient net resulted in a 68.5% accuracy!!!!!! so im going to use this for now, since this is the best result I've gotten so far.

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

BATCH_SIZE = 32

# Adjust transformations for EfficientNet expectations
TRANSFORM = Compose([
    Resize((224, 224)),  # EfficientNet expects input size of 224x224 for B0
    Grayscale(num_output_channels=3),  # Convert to 3 channels as EfficientNet uses RGB images
    RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
    RandomHorizontalFlip(),
    RandomRotation(20),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for EfficientNet
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

def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    for batch, (X, y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    running_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print(f"Training Loss: {running_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%")

def validate(dataloader, model, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    running_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    print(f"Validation Loss: {running_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%")


epochs = 11
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t)
    validate(val_dataloader, model, loss_fn)

#final eval on the test set
validate(test_dataloader, model, loss_fn)

# Save the model after training
torch.save(model.state_dict(), "efficientnet_model.pth")
print("Training complete. Model saved as efficientnet_model.pth")