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

RUNS = 3
VOCAB_SIZE = len(Main.i2w)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Movie Review Classification",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "RNN",
    "dataset": "CIFAR-100",
    "epochs": 21,
    }
)

class SimpleSelfAttention(nn.Module):
    def __init__(self, b, t, k):
        super().__init__()
        self.b = b
        self.t = t
        self.k = k

    def forward(self, x):
        b, t, k = x.size()
        # Compute raw weights
        raw_weights = torch.bmm(x, x.transpose(1, 2))
        # Apply row-wise softmax
        weights = F.softmax(raw_weights, dim=2)
        # Compute output sequence
        y = torch.bmm(weights, x)
        return y

class MultiheadSelfAttention(nn.Module):
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

        # Use assert to check the shape of the tensors

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
    
class AlternativeSelfAttention(nn.Module):
    def __init__(self, embed_size, heads = 4, mask = False):
        super(AlternativeSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
       
        self.tokeys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.toqueries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.tovalues = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.unifyheads = nn.Linear(heads * self.head_dim, embed_size)

        self.mask = mask

    def forward(self, values, keys, query, mask):
        H = self.heads
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, H, self.head_dim)
        keys = keys.reshape(N, key_len, H, self.head_dim)
        query = query.reshape(N, query_len, H, self.head_dim)

        keys = self.tokeys(keys)
        queries = self.toqueries(queries)
        values = self.tovalues(values)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        if self.mask is not None:
            energy = energy.masked_fill(self.mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.unifyheads(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadSelfAttention(embed_size, heads)
        self.attention = AlternativeSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout = dropout,
                    forward_expansion = forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = AlternativeSelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out  
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, 
                embed_size, num_layers, heads,
                forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, 
                trg_vocab_size, 
                src_pad_idx, 
                trg_pad_idx, 
                embed_size = 256, 
                num_layers = 6, 
                forward_expansion = 4, heads = 8, dropout = 0, device = 'cuda', max_length = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, 
                               embed_size, 
                               num_layers, 
                               heads, 
                               device, 
                               forward_expansion, 
                               dropout, 
                               max_length)
        self.decoder = Decoder(trg_vocab_size,
                                embed_size,
                                num_layers,
                                heads,
                                forward_expansion,
                                dropout,
                                device,
                                max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_trg_mask(self, trg):
        N, trg_length = trg.shape
        trg_mask = torch.tril(torch.ones((trg_length, trg_length))).expand(N, 1, 
                                                                           trg_length, trg_length)
        return trg_mask.to(self.device)
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

def testTransformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 100
    trg_vocab_size = 100
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    
    for (inputs, labels) in Main.train_dataset:
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(inputs.shape)
        print(labels.shape)
        outputs = model(inputs, labels)
        print(outputs.shape)
        break


class Classifier(nn.Module):
    def __init__(self, vocab_size, pool_type = 'max', output_dim = 4, embed_dim = 128):
        super(Classifier, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.pool_type = pool_type 

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.linear = nn.Linear(embed_dim, output_dim, bias=True)
        self.self_attention = MultiheadSelfAttention(output_dim)
        
    def forward(self, x): 
        x = self.embedding(x) 
        x = self.linear(x)
        x = self.self_attention(x)

        if self.pool_type == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pool_type == 'avg':
            x = torch.mean(x, dim=1)
        else:
            print("Should be max or mean pooling\n")
        return x

net = Classifier(VOCAB_SIZE, pool_type="avg")

# def runClassifier(pool_type):
#     classifier = Classifier(VOCAB_SIZE, pool_type)
#     return classifier

def optimization():
    # net = runClassifier(pool_type=any)
    return nn.CrossEntropyLoss(), optim.Adam(net.parameters(), lr=0.001)

criterion, optimizer = optimization()

start_time = time.time()

def trainInstancesClassifier():
    # net = runClassifier(pool_type=any)
    for epoch in range(RUNS):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Main.train_dataset:

            optimizer.zero_grad()

            outputs = net(inputs)
            labels = labels.view(-1)

            # print(outputs.shape, "\n", labels.shape, "\n")
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

        wandb.log({"Accuracy Instance Classifier, max pooling": epoch_accuracy, "Loss Instance Classifier: max pooling": epoch_loss}); 

        # print(f'Epoch [{epoch+1}/{RUNS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')
        # print("======================================================") 
        # print(f'Total run time: {int(minutes)}:{int(seconds)}\n')

def trainTokensClassifier():
    for epoch in range(RUNS):  # loop over the dataset multiple times
        # net = runClassifier(pool_type=any)
        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Tokens.train_dataset_by_tokens:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            labels = labels.squeeze()
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

        wandb.log({"epochs": epoch +1 /RUNS, "train token accuracy": epoch_accuracy, "train token loss": epoch_loss}); 

        # print(f'Epoch [{epoch+1}/{RUNS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%')
        # print("======================================================") 
        # print(f'Total run time: {int(minutes)}:{int(seconds)}')
    
def testClassifier():
    running_loss = 0.0
    running_accuracy = 0.0
    # net = runClassifier(pool_type=any)
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
    wandb.log({"Test accuracy Instance Classifier, max pooling": test_accuracy, "Test loss Instance Classifier, max pooling": test_loss})

def testTokensClassifier():
    running_loss = 0.0
    running_accuracy = 0.0

    # net = runClassifier(pool_type=any)
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

    wandb.log({"Test accuracy Instance Classifier, max pooling": test_accuracy, "Test loss Instance Classifier, max pooling": test_loss})
    # print(f'Test Token Loss: {test_loss:.4f}, Test Token Accuracy: {test_accuracy:.4f}%')
    
def handler(classifier): #, pool_type):
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
    
    # if pool_type == "max":
    #     print("Using max pooling")
    #     Classifier(VOCAB_SIZE, pool_type='max')
    # elif pool_type == "mean":
    #     print("Using mean pooling")
    #     Classifier(VOCAB_SIZE, pool_type='avg')
   
if __name__ == '__main__':
  fire.Fire(handler)


