import util.data_rnn as data_rnn
import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

# Sort the reviews
# def sort_reviews(i2w, x_train): #  y_train
#     return sorted(x_train, key = lambda x: sum(len(i2w[w]) for w in x))
def sort_reviews(i2w, x_train, y_train):
    sorted_reviews = sorted(zip(x_train, y_train), key=lambda pair: sum(len(i2w[w]) for w in pair[0]))
    sorted_x_train, sorted_y_train = zip(*sorted_reviews)
    return sorted_x_train, sorted_y_train

def batch_sequences_by_instance(sequence, y_train, batch_size): 
    batches = []
    sentiments = []
    for i in range(0, len(sequence), batch_size):
        batch = sequence[i:i + batch_size]
    
        batch_words = [] 
        for seq in batch:
        #    words = [i2w[idx] for idx in seq]
            batch_words.append(seq)
        batches.append(batch_words)

    for i in range(0, len(y_train), batch_size):
        sentiment_labels = y_train[i:i + batch_size]

        sentiment_to_words = []

        for seq in sentiment_labels:
            sentiment_to_words.append([seq])
        sentiments.append([sentiment_labels])

    return batches, sentiments

def batch_sequences_by_tokens(sequence, batch_size, seq_len): 
    # Sort the data by the total number of tokens
    for i in range(0, len(sequence), batch_size):
        # Batch the sequences by the maximum number of tokens
        batch = sequence[i:i+batch_size]
        current_batch = []
        current_batch_tokens = 0
        for seq in sequence:
            sequence_tokens = sum(seq)
            
            # If the current sequence exceeds the maximum number of tokens, skip it
            if sequence_tokens > seq_len:
                continue
            # If the current sequence can fit into the current batch, add it
            if current_batch_tokens + sequence_tokens <= seq_len:
                current_batch.append(seq)
                current_batch_tokens += sequence_tokens
            # If the current sequence cannot fit into the current batch, start a new batch
            else:
                batch.append(current_batch)
                current_batch = seq
                current_batch_tokens = sequence_tokens
        
        # Add the final batch if it's not empty
        if current_batch:
            batch.append(current_batch)
    return batch

def get_max(sorted_sequence):
    max_len = max(len(review) for review in sorted_sequence)
    # max_len = max(map(len, sorted_sequence))
    return max_len

def get_padded_sequence_and_labels(sequences, labels): #y_train
    seq = []
    for batch in sequences:
        max_batch = get_max(batch)
        for seq in batch:
            if max_batch is None:
                max_batch = get_max(sequences)

            if len(seq) > max_batch:
                seq = seq[:max_batch]

            # Pad sequence to max length with zeros
            seq += [0] * (max_batch - len(seq))

    for sentiment in labels:
        max_sentiment = get_max(sentiment)
        for seq in sentiment:

            if max_sentiment is None:
                max_sentiment = get_max(labels)

            if len(seq) > max_sentiment:
                seq = seq[:max_sentiment]

            # Pad sequence to max length with zeros
            seq += ([0] * (max_sentiment - len(seq)),)

    return sequences, labels

def get_review_tensor(padded_reviews):
    review_list_tensor = []
    for seq in padded_reviews:
        tensor_batch = torch.tensor(seq, dtype = torch.long)
        review_list_tensor.append(tensor_batch)
    return review_list_tensor

def get_sentiment_tensor(y_train):
    sentiment_tensors = []
    for sentiment in y_train:
        sentiment_tensor = torch.tensor(sentiment, dtype = torch.long)
        sentiment_tensors.append(sentiment_tensor)
    return sentiment_tensors

def get_train_review_tensor(x_val):
    train_review_list_tensor = []
    for review in x_val:
        review_tensor = torch.tensor(review, dtype = torch.long)
        train_review_list_tensor.append(review_tensor)
    return train_review_list_tensor

def get_train_sentiment(y_val):
    # sentiment_tensor = torch.tensor(y_train, dtype = torch.long)
    train_sentiment_list_tensor = []
    for sentiment in y_val: 
        sentiment_tensor = torch.tensor(sentiment, dtype = torch.long)
        train_sentiment_list_tensor.append(sentiment_tensor)
    return train_sentiment_list_tensor

def append_lists(review_list, sentiment_list):
    result = []
    for i in range(len(review_list)):
        result.append([review_list[i], sentiment_list[i]])
    
    return result

## TODO use fire module

## Simple self attention
## Each of the outputs is a weighted vector of the input vectors
## Sum of the square sequence lengths is a constant over the tokens







