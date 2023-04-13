import util
import torch

class Main():
    
    # Load the data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = util.data_rnn.load_imdb(final = False)

    # Sort the reviews
    sorted_x_train, sorted_y_train = util.loading.sort_reviews(i2w, x_train, y_train) #y_train

    # Batch by a number of fixed instances
    batched_x_train, batched_y_train = util.loading.batch_sequences_by_instance(sorted_x_train, sorted_y_train, batch_size = 32)
    
    # Get the maximum length for padding
    # max_len = util.loading.get_max(batched_sorted_reviews)
    
    # Pad reviewsmanually with 0's. We also pad the labels so that they match the length of the batches of reviews
    padded_reviews_x, padded_sentiments_y = util.loading.get_padded_sequence_and_labels(batched_x_train, batched_y_train) 
    
    # Build tensors

    padded_tensors = util.loading.get_review_tensor(padded_reviews_x)

    sentiment_tensors = util.loading.get_sentiment_tensor(padded_sentiments_y)
    
    print(padded_tensors, sentiment_tensors)


main = Main()








