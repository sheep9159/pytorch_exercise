import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets
from torchvision import transforms


LEARNING_RATE = 0.005
BATCH_SIZE = 16
TIME_STEP = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 64
NUM_LAYERS = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_datasets = torchvision.datasets.MNIST(
    'mnist',
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=False,
)

test_datasets = torchvision.datasets.MNIST(
    'mnist',
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=False,
)

train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
test_x = test_datasets.data.type(torch.FloatTensor)[:2000]/255
test_y = test_datasets.targets[:2000].data.numpy()
# print(type(test_datasets.data[1]))


class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.LSTM_01 = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.linear = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, input):
        input, (h_n, c_n) = self.LSTM_01(input, None)
        input = input[:, -1, :]
        input = self.linear(input)

        return input


rnn = LSTMNet().to(device)
# print(rnn)

loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.squeeze(), batch_y.squeeze()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = rnn(batch_x)
        loss = loss_func(pred, batch_y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 500 == 0:
            test_x= test_x.to(device)
            test_output = rnn(test_x)
            pred = torch.argmax(test_output, 1).cpu().data.numpy().squeeze()
            accuracy = (sum(pred == test_y) / 2000) * 100
            print('epoch:{} | step:{:4d} | loss:{:0.3f} | acc:{:0.3f}%'.format(epoch, step, loss, accuracy))

test_x = test_x[:10].to(device)
test_output = rnn(test_x)
pred = torch.argmax(test_output, 1).cpu().data.numpy().squeeze()
print(pred, '\n', test_y[:10])




