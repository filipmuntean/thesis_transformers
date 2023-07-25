import util

class Tokens():

    # Load the data
    (x_train_tokens, y_train_tokens), (x_val_tokens, y_val_tokens), (i2w_tokens, w2i_tokens), numcls = util.data_rnn.load_imdb(final = False)

    # Sort the reviews
    sorted_x_train_tokens, sorted_y_train_tokens = util.loading_tokens.sort_reviews(i2w_tokens, x_train_tokens, y_train_tokens)

    # Batch by a number of fixed tokens
    batched_x_train_by_tokens, batched_y_train_by_tokens = util.loading_tokens.batch_sequences_by_tokens(sorted_x_train_tokens, sorted_y_train_tokens, batch_size = 32, seq_len = 8388608)

    padded_reviews_x_by_tokens, padded_reviews_y_by_tokens = util.loading_tokens.get_padded_sequences_and_labels(batched_x_train_by_tokens, batched_y_train_by_tokens)

    # Build tensors
    review_tensors_by_tokens = util.loading_tokens.get_review_tensor(padded_reviews_x_by_tokens)
        
    sentiment_tensors_by_tokens = util.loading_tokens.get_sentiment_tensor(padded_reviews_y_by_tokens)
    
    # Concatenate the train tensors
    train_dataset_by_tokens = util.loading.append_lists(review_tensors_by_tokens, sentiment_tensors_by_tokens)

    # Load the test data
    sorted_x_test_tokens, sorted_y_test_tokens = util.loading_tokens.sort_reviews(i2w_tokens, x_val_tokens, y_val_tokens)

    batched_x_test_by_tokens, batched_y_test_by_tokens = util.loading_tokens.batch_sequences_by_tokens(sorted_x_test_tokens, sorted_y_test_tokens, batch_size = 32, seq_len = 8388608)

    padded_reviews_test_x_by_tokens, padded_reviews_test_y_by_tokens = util.loading_tokens.get_padded_sequences_and_labels(batched_x_test_by_tokens, batched_y_test_by_tokens)
    # Get the test tensors
    test_reviews_tokens = util.loading.get_review_tensor(padded_reviews_test_x_by_tokens)
    test_sentiments_tokens = util.loading.get_sentiment_tensor(padded_reviews_test_y_by_tokens)

    # Concatenatethe test tensors
    test_dataset_by_tokens = util.loading.append_lists(test_reviews_tokens, test_sentiments_tokens)

tokens = Tokens()
