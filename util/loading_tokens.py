import torch
import torch.nn.utils.rnn as rnn_utils

BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

# Sort the reviews
def sort_reviews(i2w, x_train, y_train):
    sorted_reviews = sorted(zip(x_train, y_train), key=lambda pair: sum(len(i2w[w]) for w in pair[0]))
    sorted_x_train, sorted_y_train = zip(*sorted_reviews)
    return sorted_x_train, sorted_y_train

def batch_sequences_by_tokens(sequence, y_train, batch_size, seq_len): 

    final_batch_seq = []
    final_batch_sent = []
    current_batch_tokens = 0

    for i in range(0, len(sequence), batch_size):
        batch_sequence = sequence[i:i+batch_size]
        current_batch_sequences = []

        for seq in batch_sequence:
            sequence_tokens = sum(seq)

            if sequence_tokens > seq_len:
                continue
            if current_batch_tokens + sequence_tokens <= seq_len:
                current_batch_sequences.append(seq)
                current_batch_tokens += sequence_tokens
            else:
                current_batch_sequences = [seq]
                current_batch_tokens = sequence_tokens
        if current_batch_sequences:
            final_batch_seq.append(current_batch_sequences)
    
    for i in range(0, len(y_train), batch_size):
        batch_sentiment = y_train[i:i+batch_size]
        current_batch_sentiments = []

        for sentiment in batch_sentiment:

            if sentiment > seq_len:
                continue
            if current_batch_tokens + sentiment <= seq_len:
                current_batch_sentiments.append([sentiment])
                current_batch_tokens += sentiment
            else:
                # final_batch_sent.append(current_batch_sentiments)
                current_batch_sentiments = sentiment
                current_batch_tokens = sentiment

        if current_batch_sentiments:
            final_batch_sent.append([current_batch_sentiments])

    return final_batch_seq, final_batch_sent

def get_max(sorted_sequence):
    return max(len(review) for review in sorted_sequence)
    
def get_padded_sequences_and_labels(sequences, labels): #y_train
    seq = []
    for batch in sequences:
        max_batch = get_max(batch)

        for seq in batch:
            # if max_batch is None:
            #     max_batch = get_max(batch)

            if len(seq) > max_batch:
                seq = seq[:max_batch]

            # Pad sequence to max length with zeros
            seq += [0] * (max_batch - len(seq))

    
    for sentiment in labels:
        max_sentiment = get_max(sentiment)
        # if max_sentiment is None:
        #     max_sentiment = get_max(sentiment)

        if len(sentiment) > max_sentiment:
            seq = seq[:max_sentiment]

        # Pad sequence to max length with zeros
        seq += ([0] * (max_sentiment - len(seq)))
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
