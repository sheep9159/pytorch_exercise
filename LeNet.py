import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
import matplotlib.pyplot as plt


BATCH_SIZE = 64
LEARNING_RATE = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


train_data = torchvision.datasets.MNIST(
    'mnist', train=True, transform=torchvision.transforms.ToTensor(), download=False
)
test_data = torchvision.datasets.MNIST(
    'mnist', train=False, transform=torchvision.transforms.ToTensor(), download=False
)


train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255
test_x = torch.unsqueeze(test_x, dim=1).to(device)
test_y = test_data.targets[:2000].data.numpy()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            # n = ((in_channels - kernel + 2 * padding) / stride) + 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # n = in_channels / 2
            nn.MaxPool2d(2)
            # batch_size * 32 * 14 * 14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # batch_size * 64 * 7 * 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # batch_size * 64 * 3 * 3  maxpool2d整除不了就省去如 7 / 2 = 3.5取 3
        )
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)  # 将平面的(即有形状的矩阵)平展成向量,63 * 3* 3 = 576维的向量
        out = self.linear(res)

        return out


convNet = Net()
# convNet.conv1[0].weight[0] = convNet.conv2[0].weight[0][0]
convNet = convNet.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convNet.parameters(), lr=LEARNING_RATE)
schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.5)

for epoch in range(3):
    for step, (batch_train_data, batch_train_label) in enumerate(train_loader):
        batch_train_label = batch_train_label.to(device)
        batch_train_data = batch_train_data.to(device)
        prediction = convNet(batch_train_data)

        loss = loss_func(prediction, batch_train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            with torch.no_grad():
                test_output = convNet(test_x)
                prediction = torch.argmax(test_output, 1).cpu().data.numpy().squeeze()
                accuracy = (sum(prediction == test_y) / 2000) * 100
                print('epoch:{} | step:{:4d} | loss:{:0.3f} | acc:{:0.3f}%'.format(epoch, step, loss, accuracy))

    schedule.step()


# def imgshow(tensor, title=None):
#     image = tensor.cpu().clone()
#     image = torchvision.transforms.ToPILImage(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)


# plt.ion()
with torch.no_grad():
    test_x = test_x[:10]
    test_output = convNet(test_x)
    pred = torch.argmax(test_output, 1).cpu().data.numpy().squeeze()
    print(pred, '\n', test_y[:10])
    # plt.figure()
    # for i in range(10):
    #     imgshow(test_x[i], str(pred[i]))

print('use time: ', time.process_time(), 's')
