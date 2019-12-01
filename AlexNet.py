import torch
import torch.nn as nn
import torch.hub as hub
import torchvision
from torch.utils.data.dataloader import DataLoader

BATCH_SIZE = 32
LEARNING_RATE = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 图片是3* 32* 32，每一batch是32
train_data = DataLoader(dataset=torchvision.datasets.CIFAR10(
    'CIFAR-10', train=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), 2),
        torchvision.transforms.ToTensor()]
    ), download=False
), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

test_data = DataLoader(dataset=torchvision.datasets.MNIST(
    'CIFAR-10', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), 2),
        torchvision.transforms.ToTensor()]
    ), download=False
), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# img = len(list(iter(train_data)))  # for test
# print(img)
# img = torchvision.transforms.ToPILImage(img)
#
# plt.ion()
#
#
# def imshow(tensor, title=None):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = torchvision.transforms.ToPILImage(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(2)
#
# plt.figure()
# imshow(img[0], title='Image')


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


model = AlexNet().to(device)
loss_fc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1):
    for step, (batch_x, batch_y) in enumerate(train_data):

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pre = model(batch_x)
        loss = loss_fc(pre, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('epoch', epoch, ' | ', 'step', step, ' | ', 'loss: ', loss)