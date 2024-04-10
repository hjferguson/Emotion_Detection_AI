#Harlan Ferguson 101133838

#Tensorflow has proved to be difficult to work with when attempting to utalize my gpu. 
#Due to frusteration, I've moved over to pytorch. In hopes of getting my website to work in time for project submission.

#attempt 1 with a basic sequential model resulted in 11% accuracy.
#attempt 2, I used similar model structure to my tensorflow model. 53% accuracy.
#tripled the epoch and got to 56% accuracy. 56% is not good enough. I'm hoping for closer to 70%.

#I need to use a pretrained model... After some research, I'm going to try resnet18 because its a popular image recognition model.
#First train with resnet, got 51% test accuracy with a 97% train accuracy. This is a sign of overfitting. womp womp

from sys import argv
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision import models


BATCH_SIZE = 32

TRANSFORM = Compose([
    Resize((48, 48)),
    Grayscale(),
    RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
    RandomHorizontalFlip(),
    RandomRotation(20),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
])
training_data = ImageFolder(
    root="../data/archive/train",
    transform=TRANSFORM
)

test_data = ImageFolder(
    root="../data/archive/test",
    transform=TRANSFORM
)


total_train_samples = len(training_data)
val_size = int(total_train_samples * 0.15) 
train_size = total_train_samples - val_size

# Split the dataset
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

#check to see if gpu is configured, if not use cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class ResNetFER(nn.Module):
    def __init__(self, num_classes=6): 
        super(ResNetFER, self).__init__()
        # Load a pre-trained ResNet
        original_model = models.resnet34(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel (grayscale) input. Resnet uses rgb
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            *list(original_model.children())[1:-1],  # Exclude original first conv layer and the original fc layer
        )
       
        self.fc = nn.Linear(512, num_classes)  
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

    
# model = NeuralNetwork().to(device)
model = ResNetFER(num_classes=6).to(device)

#parameter optimization
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)




scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)



def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0
    for batch, (X, y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    running_loss /= size
    accuracy = correct / size
    print(f"Training Loss: {running_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%")

def test(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1} Testing"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            running_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    running_loss /= size
    accuracy = correct / size
    print(f"Test Loss: {running_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%")



def evaluate(dataloader, model, loss_fn, description="Evaluation"):
    model.eval()
    size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc=f"{description}"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            running_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    running_loss /= len(dataloader)
    accuracy = correct / size
    print(f"{description} Loss: {running_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%")
    return running_loss, accuracy


epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t)
    evaluate(val_dataloader, model, loss_fn, description="Validation")  
    val_loss, val_acc = evaluate(val_dataloader, model, loss_fn, "Validation")
    scheduler.step(val_loss) 

    

evaluate(test_dataloader, model, loss_fn, description="Final Test")
torch.save(model.state_dict(), "model.pth") #56% accuracy