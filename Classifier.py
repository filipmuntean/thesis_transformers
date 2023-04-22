import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn 
import torch.nn.functional as F
from main import Main
from tokens import Tokens
import time
import fire
import wandb
import random

RUNS = 1
VOCAB_SIZE = len(Main.i2w)

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Movie Review Classification",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.001,
#     "architecture": "RNN",
#     "dataset": "CIFAR-100",
#     "epochs": 21,
#     }
# )

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 4, mask = False):
        
        super().__init__()

        assert k % heads == 0, "k must be divisible by heads"

        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias = False)
        self.toqueries = nn.Linear(k, k, bias = False)
        self.tovalues = nn.Linear(k, k, bias = False)

        self.unifyheads = nn.Linear(k, k)

        self.mask = mask

    def forward(self, x):
        B, T, K = x.size()
        H = self.heads

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        S = K // H

        keys = keys.view(B, T, H, S)
        queries = queries.view(B, T, H, S)
        values = values.view(B, T, H, S)

        keys = keys.transpose(1, 2).contiguous().view(B * H, T, S)
        queries = queries.transpose(1, 2).contiguous().view(B * H, T, S)
        values = values.transpose(1, 2).contiguous().view(B * H, T, S)

        dot = torch.bmm(queries, keys.transpose(1, 2))

        dot = dot / (K ** (1/2))

        if self.mask:
            mask = torch.triu(torch.ones(T, T), diagonal = 1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)
            dot = dot.masked_fill(mask, -float('inf'))

        dot = F.softmax(dot, dim = 2)

        out = torch.bmm(dot, values).view(B, H, T, S)
        out = out.transpose(1, 2).contiguous().view(B, T, S * H)

        return self.unifyheads(out)

class Classifier(nn.Module):
    def __init__(self, vocab_size, output_dim = 4, embed_dim = 128, pool_type = 'avg'):
        super(Classifier, self).__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pooling = pool_type 
        self.linear = nn.Linear(embed_dim, output_dim, bias=True)
        self.self_attention = SelfAttention(output_dim)
        
    def forward(self, x): 
        x = self.embedding(x) 
        x = self.linear(x)
        x = self.self_attention(x)

        if self.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling == 'avg':
            x = torch.mean(x, dim=1)
        else:
            print("Should be max or mean pooling\n")
        return x

net = Classifier(VOCAB_SIZE, pool_type='avg')

def optimization():
    return nn.CrossEntropyLoss(), optim.Adam(net.parameters(), lr=0.001)

criterion, optimizer = optimization()

start_time = time.time()

def trainInstancesClassifier():
    for epoch in range(RUNS):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Main.train_dataset:

            optimizer.zero_grad()

            outputs = net(inputs)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).sum().item() / labels.numel()
            running_accuracy += batch_accuracy

            running_loss += loss.item()

        epoch_loss = running_loss / len(Main.train_dataset)
        epoch_accuracy = running_accuracy / len(Main.train_dataset) * 100

        end_time = time.time()
        total_time_seconds = end_time - start_time
        minutes, seconds = divmod(total_time_seconds, 60)

        wandb.log({"epochs": epoch +1 /RUNS, "acc": epoch_accuracy, "loss": epoch_loss}); 

        # print(f'Epoch [{epoch+1}/{RUNS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')
        # print("======================================================") 
        # print(f'Total run time: {int(minutes)}:{int(seconds)}')

def trainTokensClassifier():
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

        # wandb.log({"epochs": epoch +1 /RUNS, "train token accuracy": epoch_accuracy, "train token loss": epoch_loss}); 

        print(f'Epoch [{epoch+1}/{RUNS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')
        print("======================================================") 
        print(f'Total run time: {int(minutes)}:{int(seconds)}')
    
def testClassifier():
    running_loss = 0.0
    running_accuracy = 0.0

    net.eval() 

    with torch.no_grad(): 
        for (inputs, labels) in Main.test_dataset:
            outputs = net(inputs)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).sum().item() / labels.numel()
            running_accuracy += batch_accuracy

            running_loss += loss.item()

    test_loss = running_loss / len(Main.test_dataset)
    test_accuracy = running_accuracy / len(Main.test_dataset) * 100
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')
    wandb.log({"test_acc": test_accuracy, "test_loss": test_loss})

def testTokensClassifier():
    running_loss = 0.0
    running_accuracy = 0.0
    net.eval() # switch to evaluation mode

    with torch.no_grad(): # turn off gradient computation
        for (inputs, labels) in Tokens.test_dataset_by_tokens:
            # Forward pass
            outputs = net(inputs)
            labels = labels.squeeze() #.squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            if outputs.size(0) != labels.size(0):
                # Pad the smaller batch to match the size of the larger batch
                if outputs.size(0) < labels.size(0):
                    outputs = nn.functional.pad(outputs, (0, 0, 0, labels.size(0) - outputs.size(0)))
                else:
                    labels = nn.functional.pad(labels, (0, outputs.size(0) - labels.size(0)))
            loss = criterion(outputs, labels)

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
            running_accuracy += batch_accuracy

            running_loss += loss.item()

    test_loss = running_loss / len( Tokens.test_dataset_by_tokens)
    test_accuracy = running_accuracy / len(Tokens.test_dataset_by_tokens) * 100

    # wandb.log({"test_acc": test_accuracy, "test_loss": test_loss})
    print(f'Test Token Loss: {test_loss:.4f}, Test Token Accuracy: {test_accuracy:.4f}%')
    
# trainInstancesClassifier()
# testClassifier()

# trainTokensClassifier()
# testTokensClassifier()

def handler(classifier):
    if classifier == "instances":
        print("Training instances classifier")
        trainInstancesClassifier()
        testClassifier()
    elif classifier == "tokens":
        print("Training tokens classifier")
        trainTokensClassifier()
        testTokensClassifier()
    else:
        print("Should be instances or tokens classifier")
   
if __name__ == '__main__':
  fire.Fire(handler)


