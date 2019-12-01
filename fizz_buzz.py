import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time

NUM_DIGITS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 50


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), 'fizz', 'buzz', 'fizzbuzz'][prediction]


def helper(i):
    print(fizz_buzz_decode(i, fizz_buzz_encode(i)))


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])[::-1]


train_x = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)]).cuda()
train_y = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)]).cuda()
# train_x = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
# train_y = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])


# model = nn.Sequential(
#             nn.Linear(NUM_DIGITS, 100),
#             nn.ReLU(),
#             nn.Linear(100, 4)
#         )

class Net(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        x = self.linear(x)

        return x


model = Net(NUM_DIGITS, 100, 4)
model = model.cuda()
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1000):
    for start in range(0, len(train_x), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_x = train_x[start:end]
        batch_y = train_y[start:end]
        pred_y = model(batch_x)
        loss = loss_func(pred_y, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # if epoch % 50 == 0:
    #     print('epoch:', epoch, '|', 'loss:', loss)

test_x = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 100)]).cuda()
test_y = torch.LongTensor([fizz_buzz_encode(i) for i in range(1, 100)]).cuda()
# test_x = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 100)])
# test_y = torch.LongTensor([fizz_buzz_encode(i) for i in range(1, 100)])
with torch.no_grad():
    prediction = model(test_x)
    prediction = torch.max(prediction, 1)[1].cpu().data.tolist()
    # prediction = torch.max(prediction, 1)[1].data.tolist()
    # print(prediction)
    prediction = zip(range(1, 100), prediction)
    print([fizz_buzz_decode(i, item) for i, item in prediction])

actual =[]
for i in range(1, 100):
    actual.append(fizz_buzz_encode(i))

actual = zip(range(1, 100), actual)
print([fizz_buzz_decode(i, item) for i, item in actual])

print('\nElapsed time:', time.process_time())

