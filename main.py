import util
import torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

class Main():
    # Load the data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = util.data_rnn.load_imdb(final = False)
    
    # Sort the reviews
    sorted_x_train, sorted_y_train = util.loading.sort_reviews(i2w, x_train, y_train) #y_train

    # print(f"This is the sorted list of reviews{sorted_x_train[0]} and the sorted list of sentiments{sorted_y_train[0]}\n")
    # Batch by a number of fixed instances
    batched_x_train, batched_y_train = util.loading.batch_sequences_by_instance(sorted_x_train, sorted_y_train, batch_size = 32)
    # print(f"This is the list of batches reviews{batched_x_train[0]} and the list of batches sentiments{batched_y_train}\n")
    
    # Pad reviewsmanually with 0's. We also pad the labels so that they match the length of the batches of reviews
    padded_reviews_x, padded_sentiments_y = util.loading.get_padded_sequence_and_labels(batched_x_train, batched_y_train) 
    # print(f"This is the list of padded reviews{padded_reviews_x[0]} and the list of padded sentiments{padded_sentiments_y[0]}\n")
    # Build tensors

    padded_tensors = util.loading.get_review_tensor(padded_reviews_x)

    sentiment_tensors = util.loading.get_sentiment_tensor(padded_sentiments_y)
    
main = Main()

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(32, 2514, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in enumerate(main.padded_tensors, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(main.padded_tensors)
        loss = criterion(outputs, main.sentiment_tensors)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 32 == 31:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 32:.3f}')
            running_loss = 0.0


print('Finished Training')

dataiter = iter(main.x_val)
images, labels = next(dataiter)


