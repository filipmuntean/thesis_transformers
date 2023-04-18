import torch
import torch.nn as nn
import torch.optim as optim
from tokens import Tokens
import time

RUNS = 15
VOCAB_SIZE = len(Tokens.i2w_tokens)

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

net = GlobalPoolingClassifier(VOCAB_SIZE, pool_type='avg')

def optimization():
    return nn.CrossEntropyLoss(), optim.Adam(net.parameters(), lr=0.001)

criterion, optimizer = optimization()

start_time = time.time()

def trainClassifier():
    for epoch in range(RUNS):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Tokens.train_dataset_by_tokens:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            labels = labels.squeeze()
            # print(outputs.shape, "\n", labels.shape)
            if outputs.size(0) != labels.size(0):
                # Pad the smaller batch to match the size of the larger batch
                if outputs.size(0) < labels.size(0):
                    outputs = nn.functional.pad(outputs, (0, 0, 0, labels.size(0) - outputs.size(0)))
                else:
                    labels = nn.functional.pad(labels, (0, outputs.size(0) - labels.size(0)))
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).sum().item() / labels.numel()
            running_accuracy += batch_accuracy

            running_loss += loss.item()

        epoch_loss = running_loss / len(Tokens.train_dataset_by_tokens)
        epoch_accuracy = running_accuracy / len(Tokens.train_dataset_by_tokens) * 100

        end_time = time.time()
        total_time_seconds = end_time - start_time
        minutes, seconds = divmod(total_time_seconds, 60)
        print(f'Epoch [{epoch+1}/{RUNS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')
        print("======================================================") 
        print(f'Total run time: {int(minutes)}:{int(seconds)}')

trainClassifier()

def testClassifier():
    running_loss = 0.0
    running_accuracy = 0.0
    net.eval() # switch to evaluation mode

    with torch.no_grad(): # turn off gradient computation
        for (inputs, labels) in Tokens.test_dataset_by_tokens:
            # Forward pass
            outputs = net(inputs)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = criterion(outputs, labels)

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
            running_accuracy += batch_accuracy

            running_loss += loss.item()

    test_loss = running_loss / len( Tokens.test_dataset_by_tokens)
    test_accuracy = running_accuracy / len(Tokens.test_dataset_by_tokens) * 100

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')
