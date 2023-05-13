import fire
import time
import wandb
import torch
import tqdm
import random
from main import Main
import torch.nn as nn
import torch.nn as optim
from tokens import Tokens
import torch.nn.functional as F
from Transformer import transformer
from torch.nn.utils.rnn import pad_sequence
from basicTransformer import basictransformer
from Classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = 3 
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

# def testTransformer():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     src_pad_idx = 0
#     trg_pad_idx = 0
#     src_vocab_size = 100
#     trg_vocab_size = 100
#     model = transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

#     for (src, trg) in enumerate(util.Main.train_dataset):
#         trg_flat = list(itertools.chain(*trg))
#         src_tensor = torch.tensor(src, dtype=torch.long).to(device)
#         trg_tensor = torch.tensor(trg_flat, dtype=torch.long).to(device)
#         out = model(src_tensor, trg_tensor[:-1])
#         print(out.shape)

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch_vectorized():
    data = Main.review_tensors
    batch = random.choice(data)
    tensor_size = batch.size(0)
    ix = torch.randint(tensor_size - block_size, (batch_size,))

    for batch in data[:batch_size + 1]:
        reviews = torch.cat([tensor for tensor in batch])
        input = torch.stack([reviews[i:i+block_size] for i in ix])
        target = torch.stack([reviews[i+1:i+block_size+1] for i in ix])
    return input, target

def get_batch_unvectorized():
    data = Main.review_tensors
    batch = random.choice(data)
    tensor_size = batch.size(0)
    ix = torch.randint(tensor_size - block_size, (batch_size,))

    for review in data[:batch_size + 1]:
        for i in range(len(review)-1):
            input = torch.stack([review[i:i+block_size] for i in ix])
            target = torch.stack([review[i+1:i+block_size+1] for i in ix])
    input = torch.stack([review[ix[i]:ix[i] + block_size] for i, review in enumerate(batch)])
    target = torch.stack([review[ix[i] + 1:ix[i] + block_size + 1] for i, review in enumerate(batch)])
    return input, target

# xb, yb = get_batch_vectorized()
# print('inputs:')
# print(xb.shape, xb)
# print('\ntargets:')
# print(yb.shape, yb)
# print('-------')

# class BigramLanguageModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim = 8):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
    
#     def forward(self, idx, targets):
# #         if torch.max(idx) >= self.token_embedding_table.num_embeddings:
# #             print(torch.max(idx))
# #             raise ValueError("Input tensor has values outside the range of the embedding table.")
        
# #         logits = self.token_embedding_table(idx)

# #         B, T, C = logits.shape
# #         logits = logits.view(B * T, C)
# #         targets = targets.view(B*T)
# #         loss = F.cross_entropy(logits, targets)
        
# #         return logits, loss
    
# #     # def generate(self, idx, max_new_tokens):
# #     #     for _ in range(max_new_tokens):
# #     #         logits, loss = self(idx)
# #     #         logits = logits[:, -1, :]
# #     #         probs = F.softmax(logits, dim=-1)
# #     #         idx_next = torch.multinomial(probs, num_samples=1)

# #     #         idx = torch.cat((idx, idx_next), dim=1)
# #     #     return idx

# # m = BigramLanguageModel(tensor_size)
# # out = m(xb, yb)
# # print(out.shape)
# def exemplify():
#     block_size = 8
#     # x = util.Main.review_tensors[:block_size]
#     # y = util.Main.review_tensors[1:block_size + 1]

#     review_counter = 0
#     for tensor in Main.review_tensors[:block_size]:
#         for review in tensor:
#             print(review)
#             # if review_counter == 33:
#             #     for i in range(len(review)-1):
#             #         input_tokens = review[:i+1]
#             #         target_token = review[i+1]
#             #         print(f"When input is tensor {input_tokens}, the target is: {target_token}")
#             #         if review[i] == 0 and review[i+1] == 0:
#             #             break  # stop looping through this review
#             #     # Increment review_counter only after all tokens in the review have been printed
#             # review_counter += 1
# # exemplify()
# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t + 1]
#         target = yb[b, t]
#         print(f"when input is {context} the target is {target}")
# net = basictransformer(k=4, num_tokens = VOCAB_SIZE)

def optimization(net):
    return nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters(), lr = 0.0001)

start_time = time.time()
# criterion, optimizer = optimization(net)

def trainTransformerInstanceClassifier(net, criterion, optimizer):
    for epoch in range(RUNS): #tqdm.trange(RUNS):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Main.train_dataset:
        # for inputs in Main.review_tensors:
            optimizer.zero_grad()
            outputs = net(inputs)
            if torch.cuda.is_available():
                outputs.cuda()
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

        # wandb.log({"Accuracy Instance Classifier, max pooling": epoch_accuracy, "Loss Instance Classifier: max pooling": epoch_loss}); 

        print("Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%".format(epoch+1, RUNS, epoch_loss, epoch_accuracy))
        print("======================================================") 
        print("Total run time: {}:{}".format(int(minutes), int(seconds)))

def testTransformerClassifier(net, criterion, optimizer):
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
    print("Test Loss: {:.4f}, Test Accuracy: {:.4f}%".format(test_loss, test_accuracy))
    # wandb.log({"Test accuracy Instance Classifier, max pooling": test_accuracy, "Test loss Instance Classifier, max pooling": test_loss})

def trainTokensClassifier(net, criterion, optimizer):
    for epoch in range(RUNS):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Main.train_dataset:
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

        # wandb.log({"epochs": epoch +1 /RUNS, "train token accuracy": epoch_accuracy, "train token loss": epoch_loss}); 

        print("Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%".format(epoch+1, RUNS, epoch_loss, epoch_accuracy))
        print("======================================================") 
        print("Total run time: {}:{}".format(int(minutes), int(seconds)))

def testTransformerTokensClassifier(net, criterion, optimizer):
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

    # wandb.log({"Test accuracy Instance Classifier, max pooling": test_accuracy, "Test loss Instance Classifier, max pooling": test_loss})
    print("Test Loss: {:.4f}, Test Accuracy: {:.4f}%".format(test_loss, test_accuracy))

def trainFullTransformerInstances(net, criterion, optimizer):
    for epoch in range(RUNS): 

        running_loss = 0.0
        running_accuracy = 0.0
        for (inputs, labels) in Main.train_dataset:
        # for inputs in Main.review_tensors:
            optimizer.zero_grad()
            trg = inputs[:, :-1]
            outputs = net(inputs, trg)

            if torch.cuda.is_available():
                outputs.cuda()
            # labels = labels.view(-1)

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

        # wandb.log({"Accuracy Instance Classifier, max pooling": epoch_accuracy, "Loss Instance Classifier: max pooling": epoch_loss}); 

        print("Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%".format(epoch+1, RUNS, epoch_loss, epoch_accuracy))
        print("======================================================") 
        print("Total run time: {}:{}".format(int(minutes), int(seconds)))

def testFullTransformerInstances(net, criterion, optimizer):
    running_loss = 0.0
    running_accuracy = 0.0
    net.eval() 

    with torch.no_grad(): 
        for (inputs, labels) in Main.test_dataset:
            trg = inputs[:, :-1]
            outputs = net(inputs, trg)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).sum().item() / labels.numel()
            running_accuracy += batch_accuracy

            running_loss += loss.item()

    test_loss = running_loss / len(Main.test_dataset)
    test_accuracy = running_accuracy / len(Main.test_dataset) * 100
    print("Test Loss: {:.4f}, Test Accuracy: {:.4f}%".format(test_loss, test_accuracy))
    
class Handler(object):

    def __init__(self, classifier="classifier", batch="instances"):
        super().__init__()
        # self.pool_type = pool_type
        self.classifier = classifier
        self.batch = batch

    @staticmethod
    def run(batch="instances", classifier = "classifier"):
        
        if classifier == "classifier":
            print("Using the normal Classifier in the GENERATOR.py file")
            net = Classifier(VOCAB_SIZE, pool_type='max')
        elif classifier == "basic":
            print("Using the basic transformer in the GENERATOR.py file")
            net = basictransformer(num_tokens = VOCAB_SIZE, k = 128, num_classes = 4, heads = 2, depth = 6, seq_length = 16384)
        elif classifier == "transformer":
            print("Using the big transformer in the GENERATOR.py file")
            net = transformer(src_vocab_size = VOCAB_SIZE, trg_vocab_size = VOCAB_SIZE, src_pad_idx=0, trg_pad_idx=0, device=device)
        else: 
            print("Should be classifier, basic or transformer")
            return 1
        
        # if k == 128:
        # elif k == 256:
        #     net = basictransformer(k=256, num_tokens = VOCAB_SIZE)
        # elif k == 512:
        #     net == basictransformer(k=512, num_tokens = VOCAB_SIZE)
        # else:
        #     print("k should be 128, 256 or 512")
        #     return 1 
        criterion, optimizer = optimization(net)

        if batch == "instances" and (classifier == "classifier" or classifier == "basic"):
            print("Training instances classifier in the GENERATOR.py file")
            trainTransformerInstanceClassifier(net, criterion, optimizer)
            testTransformerClassifier(net, criterion, optimizer)
        elif batch == "tokens" and (classifier == "classifier" or classifier == "basic"):
            print("Training tokens classifier GENERATOR.py file")
            trainTokensClassifier(net, criterion, optimizer)
            testTransformerTokensClassifier(net, criterion, optimizer)
        elif batch == "instances" and classifier == "transformer":
            print("Training BIG BOI transformer in the GENERATOR.py file")
            trainFullTransformerInstances(net, criterion, optimizer)
            testFullTransformerInstances(net, criterion, optimizer)
        else:
            print("Should be instances or tokens classifier")
   
if __name__ == '__main__':
  fire.Fire(Handler)
# for i in range(num_epochs):
#     # generate a batch of training examples
#     input_batch, target_batch = sample_batch(data, length=4, batch_size=8)

#         # forward pass, backward pass, and update weights here
#     # # Step 4: Create an instance of the transformer and move it to the GPU
#     # device = torch.device("meta" if torch.cuda.is_available() else "cpu")
#     # transformer = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.Adam(transformer.parameters(), lr=0.001)

#     # # Step 5: Train the transformer
#     # for epoch in range(epochs):
        
#     #         src = src.to(device)
#     #         trg = trg.to(device)

#     #         optimizer.zero_grad()

#     #         output = transformer(src, trg[:, :-1])
#     #         loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

#     #         loss.backward()
#     #         optimizer.step()

#     # # Step 6: Evaluate the transformer on the test set
#     # transformer.eval()
#     # with torch.no_grad():
#     #     for i, (src, trg) in enumerate(Main.test_dataset):
#     #         src = src.to(device)
#     #         trg = trg.to(device)

#     #         output = transformer(src, trg[:, :-1])
#     #         loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

#     # # Step 7: Save the model
#     # torch.save(transformer.state_dict(), "transformer.pth")


        

