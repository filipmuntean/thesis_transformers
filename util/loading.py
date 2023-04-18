import torch
import torch.nn.utils.rnn as rnn_utils

BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

# Sort the reviews

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
            batch_words.append(seq)
        batches.append(batch_words)

    for i in range(0, len(y_train), batch_size):
        sentiment_labels = y_train[i:i + batch_size]
        sentiment_to_words = []

        for seq in sentiment_labels:
            sentiment_to_words.append([seq])
        sentiments.append([sentiment_labels])

    return batches, sentiments

def get_max(sorted_sequence):
    return max(len(review) for review in sorted_sequence)
    
def get_padded_sequences_and_labels(sequences, labels): #y_train
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

            # Pad labels to max length with zeros
            seq += ([0] * (max_sentiment - len(seq)),)

    return sequences, labels

def get_review_tensor(padded_reviews):
    review_list_tensor = []
    for sequence in padded_reviews:
        tensor_batch = torch.tensor(sequence, dtype = torch.long)
        review_list_tensor.append(tensor_batch)
    return review_list_tensor

def get_sentiment_tensor(padded_sentiments):
    sentiment_tensors = []
    for sentiment in padded_sentiments:
        sentiment_tensor = torch.tensor(sentiment, dtype = torch.long)
        sentiment_tensors.append(sentiment_tensor)
    return sentiment_tensors

def append_lists(review_list, sentiment_list):
    result = []
    for i in range(len(review_list)):
        result.append([review_list[i], sentiment_list[i]])
    return result

## TODO use fire module

## Simple self attention
## Each of the outputs is a weighted vector of the input vectors
## Sum of the square sequence lengths is a constant over the tokens
