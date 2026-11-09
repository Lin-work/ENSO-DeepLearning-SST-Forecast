import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
import netCDF4 as nc

sst_data = nc.Dataset("sst.mon.mean.trefadj.anom.1880to2018.nc")
print(sst_data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
        )
        self.fc1 = nn.Linear(in_features=64*15*30, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

def replace_non_numeric(arr):
    flat_arr = arr.reshape(-1)
    flat_series = pd.Series(flat_arr)
    flat_series = pd.to_numeric(flat_series, errors='coerce').fillna(0)
    return flat_series.values.reshape(arr.shape)

def train(model, x, y, optimizer):
    model.train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    output = model(x)
    output = torch.squeeze(output)
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, y):
    model.eval()
    x, y = x.to(device), y.to(device)
    output = model(x)
    output = torch.squeeze(output)
    loss = F.mse_loss(output, y)
    return loss.item(), output.cpu().detach().numpy()


# Call input data
nino34_index = (np.loadtxt("nino34.long.anom.data.txt",skiprows=90,max_rows=139,usecols=(range(1,13))).reshape(-1))[:696]
sst_data = replace_non_numeric(sst_data)

# Normalize data
sst_data = (sst_data - sst_data.mean()) / sst_data.std()

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(sst_data[:552], nino34_index[:552], test_size=0.99, random_state=42)
x_pred = sst_data[564:]
y_pred = nino34_index[564:]

# Convert to tensors
x_train = torch.from_numpy(x_train).float().unsqueeze(1)  # Add channel dimension
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float().unsqueeze(1)  # Add channel dimension
y_test = torch.from_numpy(y_test).float()
x_pred = torch.from_numpy(x_pred).float().unsqueeze(1)  # Add channel dimension
y_pred = torch.from_numpy(y_pred).float()

print('x_train:', x_train.shape, 'y_train:', y_train.shape)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)

# Create data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size,
    shuffle=False
)

pred_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_pred, y_pred),
    batch_size=batch_size,
    shuffle=False
)

# Model setting
model_cnn = CNN()
model_cnn.to(device)

# Optimizer
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

EPOCHS = 100
Loss = np.zeros((EPOCHS, 2))

# plot the figure_1
for epoch in range(EPOCHS):
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        loss = train(model_cnn, x_batch, y_batch, optimizer)
        train_loss += loss
    train_loss /= len(train_loader)

    test_loss = 0.0
    for x_batch, y_batch in test_loader:
        loss, _ = test(model_cnn, x_batch, y_batch)
        test_loss += loss
    test_loss /= len(test_loader)

    print(f'Epoch {epoch + 1}')
    print(f'     Training set - loss: {train_loss}')
    print(f'     Validation set - loss: {test_loss}')

    Loss[epoch, 0], Loss[epoch, 1] = train_loss, test_loss

    torch.cuda.empty_cache()
    gc.collect()

plt.plot(np.arange(1, EPOCHS + 1, 1), Loss[:, 0], linewidth = 1)
plt.plot(np.arange(1, EPOCHS + 1, 1), Loss[:, 1], linewidth = 1)
plt.xlim([0, EPOCHS])
plt.xticks(np.linspace(0, EPOCHS, 6), fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.legend(['training', 'validation'], loc = 'upper right', fontsize = 10)
plt.title('Epoch vs. Loss', fontsize = 14)
plt.savefig('HW9_4_1.png', dpi = 300)
plt.show()

plt.plot(np.arange(1, EPOCHS + 1, 1), Loss[:, 0], linewidth = 1)
plt.plot(np.arange(1, EPOCHS + 1, 1), Loss[:, 1], linewidth = 1)
plt.xlim([0, EPOCHS])
plt.xticks(np.linspace(0, EPOCHS, 6), fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.yscale('log')
plt.yticks(fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.legend(['training', 'validation'], loc = 'upper right', fontsize = 10)
plt.title('Epoch vs. Loss (Logarithmic Scale)', fontsize = 14)
plt.savefig('HW9_4_2.png', dpi = 300)
plt.show()


#===============================================================
# Make predictions
predicted_nino34 = []
actual_nino34 = []

for x_batch, y_batch in pred_loader:
    _, output = test(model_cnn, x_batch, y_batch)
    predicted_nino34.extend(output)
    actual_nino34.extend(y_batch.cpu().numpy())

# Convert to numpy arrays
predicted_nino34 = np.array(predicted_nino34)
actual_nino34 = np.array(actual_nino34)

plt.plot(predicted_nino34, linewidth = 1)
plt.plot(actual_nino34, linewidth = 1)
plt.xlim([0, 131])
plt.xticks(np.linspace(0, 120, 11), ['2007/1','2008/1','2009/1','2010/1','2011/1','2012/1','2013/1','2014/1','2015/1','2016/1','2017/1'], rotation=45)
plt.yticks(fontsize = 12)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Nino3.4 Index', fontsize = 12)
plt.legend(['predicted Nino3.4', 'actual Nino3.4'], loc = 'upper left', fontsize = 10)
plt.title('Predicted vs Actual Nino3.4 Index', fontsize = 14)
plt.savefig('HW9_4_3.png', dpi = 300)
plt.show()

#===============================================================
# Calculate prediction accuracy
correlation = np.corrcoef(predicted_nino34, actual_nino34)[0, 1]
print(f'Correlation coefficient: {correlation}')

accuracy = np.mean((predicted_nino34 - actual_nino34)**2)
print(f'Prediction MSE: {accuracy}')
