import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools


'''
Neural Network Model
...needs work :(
'''



labels = np.load('labels.npy') # 19436 labels
data = np.load('data.npy')     # 19436 preprocessed images 
                               # of x,y,z at 21 landmarks a hand

data = data.reshape(data.shape[0], 21*3)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).type(torch.LongTensor)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Model, self).__init__()
        layers = [input_dim] + hidden_dims + [output_dim]
        layer = []
        for i in range(len(layers)-1):
            layer.append(nn.Linear(layers[i], layers[i+1]))
            layer.append(nn.ReLU())
        layer = layer[:-1]
        self.seq = nn.Sequential(*layer)

    def forward(self, x):
        x = self.seq(x)
        x = torch.sigmoid(x)
        return x



batch_size = 64

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

input_dim = 21 * 3
hidden_dims = [200, 100]
output_dim = 29

model = Model(input_dim, hidden_dims, output_dim)

lr = 0.1
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num_epochs = 100
loss_values = []

for epoch in range(num_epochs):
    for X, y in train_dataloader:
        optimizer.zero_grad()
       
        pred = model(X)
        l = loss(pred, y)
        loss_values.append(l.data)
        l.backward()
        optimizer.step()
        

print("Training Complete")

step = np.linspace(0, 100, 24300)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

correct = 0
total = 0

with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        total += y.size(0)
        # correct += np.array([predicted == y.numpy()]).sum().item()
        correct += 1 if predicted == y else 0 #what the fuck

print(f'Accuracy of the network on the {total} test instances: {100 * correct // total}%')