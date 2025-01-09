# importing important data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# importing traing and testing data
train_data = datasets.FashionMNIST("./dataset",transform=transforms.ToTensor(), train=True , download=True)
test_data = datasets.FashionMNIST("./dataset",transform=transforms.ToTensor(), train=False , download=False)

# dividing data into batches for preventing complexity 
batch_size = 70
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Creating a basic neural network 
class NeuralNetwork(nn.Module):
    def __init__(self,n_hidden_1, n_hidden_2, n_hidden_3, outdim=10):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_1,n_hidden_2,
                      nn.ReLU(True))
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_2,n_hidden_3,
                      nn.ReLU(True))
        )

        self.layer3 = nn.Linear(n_hidden_3,outdim)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

n_hidden_1 = 784
n_hidden_2 = 300
n_hidden_3 = 100

model = NeuralNetwork(n_hidden_1,n_hidden_2,n_hidden_3)

# switching model to preferable device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# creating optimizer and criteration
learning_rate = learning_rate = 1e-3
criteration = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training data loop
epochs = 8
for epoch in range(epochs):
    model.train()
    accuracy_rate = 0.0
    loss_rate = 0.0
    for i, data in enumerate(train_loader):
        image,label = data
        image = image.view(image.size(0), -1)
        image = image.to(device)
        label = label.to(device)
        out = model(image)
        loss = criteration(out,label)
        loss_rate += loss.item()
        _,pred = torch.max(out,-1)
        accuracy_rate += (pred==label).float().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# testing some sample images to our model
model.eval()
loss_done_by_model = 0.0
accuracy_rate = 0.0

for data in test_loader:
    image,label = data
    image = image.to(device)
    image = image.view(image.size(0), -1)
    label = label.to(device)
    out = model(image)
    with torch.no_grad(): 
        loss = criteration(out,label)
        loss_done_by_model +=loss.item()
        _,pred = torch.max(out,1)
        accuracy_rate += (pred==label).float().mean()
        print(f"loss while testing : {loss_done_by_model/len(test_loader)} , Accuracy of model : {accuracy_rate/len(train_loader)} ")