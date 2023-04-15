import util
import torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import transformers
import time
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader

# Load the data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = util.data_rnn.load_imdb(final = False)

# Sort the reviews
sorted_x_train, sorted_y_train = util.loading.sort_reviews(i2w, x_train, y_train) #y_train

# print(sorted_x_train, sorted_y_train)

# Batch by a number of fixed instances
batched_x_train, batched_y_train = util.loading.batch_sequences_by_instance(sorted_x_train, sorted_y_train, batch_size = 32)

# Pad reviews manually with 0's. We also pad the labels so that they match the length of the batches of reviews
padded_reviews_x, padded_sentiments_y = util.loading.get_padded_sequence_and_labels(batched_x_train, batched_y_train) 

# Build tensors
review_tensors = util.loading.get_review_tensor(padded_reviews_x)
sentiment_tensors = util.loading.get_sentiment_tensor(padded_sentiments_y)

# Concatenate the tensors

test_reviews = util.loading.get_train_review_tensor(x_val)
test_sentiments = util.loading.get_train_sentiment(y_val)

train_dataset = util.loading.append_lists(review_tensors, sentiment_tensors)

# padded_tensors = padded_tensors.view(1, -1)
# sentiment_tensors = torch.transpose(sentiment_tensors, 0, 1)

# train_x_dataset = TensorDataset(padded_tensors, sentiment_tensors)

# train_dataset = DataLoader(train_x_dataset, batch_size = 32, shuffle=True)

# test_x = TensorDataset(test_reviews)
# test_y = TensorDataset(test_sentiments)

# data_loader_x = DataLoader(test_x, batch_size=32, shuffle=True)
# data_loader_y = DataLoader(test_y, batch_size=32, shuffle=True)
    
# main = Main()


 