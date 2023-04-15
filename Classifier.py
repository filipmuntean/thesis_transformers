import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from main import train_dataset, i2w
import time

RUNS = 21
VOCAB_SIZE = len(i2w)

class GlobalPoolingClassifier(nn.Module):
    def __init__(self, vocab_size, output_dim = 2, embed_dim = 128, pool_type='avg'):
        super(GlobalPoolingClassifier, self).__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pooling = pool_type 
        self.linear = nn.Linear(embed_dim, output_dim, bias=True)
        
    def forward(self, x): 
        x = self.embedding(x) # embedded: (batch_size, seq_len, embedding_dim)
        if self.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling == 'avg':
            x = torch.mean(x, dim=1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg'")
        return x
    
# unq = padded_tensors.unique(return_counts=True)

net = GlobalPoolingClassifier(VOCAB_SIZE, pool_type='avg')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

start_time = time.time()

for epoch in range(RUNS):  # loop over the dataset multiple times

    running_loss = 0.0
    running_accuracy = 0.0

    total_steps = len(train_dataset)

    for (inputs, labels) in train_dataset:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        labels = labels.view(-1)
        # labels = labels.squeeze(dim=1)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
        # Compute accuracy
        _, predicted = torch.max(outputs.data, 1)
        batch_accuracy = (predicted == labels).sum().item() / labels.numel()
        running_accuracy += batch_accuracy

        running_loss += loss.item()

        # if (i+1) % 1 == 0:
        #     print(f'Epoch [{epoch+1}/{RUNS}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = running_accuracy / len(train_dataset) * 100

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Epoch [{epoch+1}/{RUNS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')
    print("============================================") 
    print(f'Total run time: {total_time:.2f} seconds')



    