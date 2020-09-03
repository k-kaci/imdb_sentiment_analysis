import io
import torch
import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn import metrics

import config
import dataset
import engine
import lstm

def load_vectors(fname):
    """
    This function load  pre-trained word vectors
    :param fname: filename
    :return: a python dictionary with embedding vectors
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix


def run(df, fold):
    """
    Run training and validation for a given fold and dataset
    :param df:  pandas dataframe with kfold column
    :param fold: current fold, int
    """
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print("Fitting tokenizer")
    # we use tf.keras for tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    X_train = tokenizer.texts_to_sequences(train_df.review.values)
    X_valid = tokenizer.texts_to_sequences(valid_df.review.values)

    # zero pad the training/validation sequences
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=config.MAX_LEN)
    X_valid = tf.keras.preprocessing.sequence.pad_sequences(X_valid, maxlen=config.MAX_LEN)

    # initialize dataset class for training/validation
    train_dataset = dataset.IMDBDataset(reviews=X_train,
                                        targets=train_df.sentiment.values)
    valid_dataset = dataset.IMDBDataset(reviews=X_valid,
                                        targets=valid_df.sentiment.values)

    # create torch dataloader for training/validation
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config.TRAIN_BATCH_SIZE,
                                                    num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=config.VALID_BATCH_SIZE,
                                                    num_workers=1)

    print("Loading embeddings")
    embedding_dict = load_vectors("../input/wiki-news-300d-1M.vec")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index,
                                               embedding_dict)
    # create torch device, since we use gpu, we are using cuda
    device = torch.device("cuda")

    # fetch our LSTM model
    model = lstm.LSTM(embedding_matrix)

    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Model")
    # set best accuracy to zero best_accuracy = 0
    # set early stopping counter to zero early_stopping_counter = 0
    # train and validate for all epochs
    best_accuracy = 0
    early_stopping_counter = 0

    for epoch in range(config.EPOCHS):
    # train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outputs, targets = engine.evaluate(
            valid_data_loader, model, device)
        # use threshold of 0.5
        outputs = np.array(outputs) >= 0.5

        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"FOLD:{fold}, Epoch: {epoch}, "
              f"Accuracy Score = {accuracy}")

        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2: break



if __name__ == "__main__":
    df = pd.read_csv("../input/imdb_folds.csv")
    # train for all folds
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)













