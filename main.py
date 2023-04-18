import util

class Main():

    # Load the data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = util.data_rnn.load_imdb(final = False)

    # Sort the reviews
    sorted_x_train, sorted_y_train = util.loading.sort_reviews(i2w, x_train, y_train)

    # Batch by a number of fixed instances
    batched_x_train, batched_y_train = util.loading.batch_sequences_by_instance(sorted_x_train, sorted_y_train, batch_size = 32)

    # Pad reviews manually with 0's. We also pad the labels so that they match the length of the batches of reviews
    padded_reviews_x, padded_sentiments_y = util.loading.get_padded_sequences_and_labels(batched_x_train, batched_y_train) 

    # Build tensors
    review_tensors = util.loading.get_review_tensor(padded_reviews_x)
    
    sentiment_tensors = util.loading.get_sentiment_tensor(padded_sentiments_y)
    
    train_dataset = util.loading.append_lists(review_tensors, sentiment_tensors)

    sorted_x_val, sorted_y_val = util.loading.sort_reviews(i2w, x_val, y_val)

    batched_x_val, batched_y_val = util.loading.batch_sequences_by_instance(sorted_x_val, sorted_y_val, batch_size = 32)

    padded_reviews_test_x, padded_sentiments_test_y = util.loading.get_padded_sequences_and_labels(batched_x_val, batched_y_val)
    # Concatenate the test tensors
    test_reviews = util.loading.get_review_tensor(padded_reviews_test_x)
    
    test_sentiments = util.loading.get_sentiment_tensor(padded_sentiments_test_y)

    test_dataset = util.loading.append_lists(test_reviews, test_sentiments)

main = Main()




 