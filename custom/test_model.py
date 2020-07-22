import os
import glob
from pathlib import Path

from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from custom.model import LSTMClassifier
from sklearn.model_selection import train_test_split

def read_imdb_data(data_dir='./data/aclImdb'):
    """
    This function create two dictionaries, one with the reviews, and other
    with the targets
    :param data_dir:
    :return:
    """
    data = {}
    labels = {}
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        # define the folder path
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)
    return data, labels


def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    # Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    # Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    text = BeautifulSoup(review, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Convert to lower case
    words = text.split()  # Split string into words
    words = [w for w in words if w not in stopwords.words("english")]  # Remove stopwords
    words = [stemmer.stem(w) for w in words]  # stem
    return words


cache_dir = os.path.join("./cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists
def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        # words_train = list(map(review_to_words, data_train))
        # words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]

        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                              cache_data['words_test'],
                                                              cache_data['labels_train'],
                                                              cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test


def build_dict(data, vocab_size=5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    word_counts = Counter(np.concatenate(data, axis=0))
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    print(len(sorted_words))
    word_dict = {}  # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]):  # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2  # 'infrequent' labels
    return word_dict


def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0  # We will use 0 to represent the 'no word' category
    INFREQ = 1  # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    working_sentence = [NOWORD] * pad
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
    return working_sentence, min(len(sentence), pad)


def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
    return np.array(result), np.array(lengths)


def train(model, train_loader,val_loader, epochs, optimizer, loss_fn, device):
    valid_loss_min = np.Inf
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch_X, batch_y = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            model.zero_grad()
            # get predictions
            output = model(batch_X)
            # compute the loss
            loss = loss_fn(output, batch_y)
            # get the loss gradients (chain rule)
            loss.backward()
            # optimize parameters
            optimizer.step()
            #train_loss += loss.data.item()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, batch in enumerate(val_loader):
            # move to GPU
            batch_X, batch_y = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch_X)
            # calculate the batch loss
            loss = loss_fn(output, batch_y)
            # update average validation loss
            #valid_loss += loss.data.item()
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,# / len(train_loader),
            valid_loss #/ len(val_loader)
        ))
        # print("Epoch: {}, BCELoss: {}".format(epoch, train_loss / len(train_loader)))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), "model.pt")
            valid_loss_min = valid_loss

    torch.save(model.state_dict(), "model_last.pt")




def predict(model, test_review, device="cpu"):
    test_review_words = review_to_words(test_review)
    test_data, test_data_len = convert_and_pad(word_dict, test_review_words)
    data_pack = np.hstack((test_data_len, test_data))
    data_pack = data_pack.reshape(1, -1)
    data_tensor = torch.from_numpy(data_pack)
    data_tensor = data_tensor.to(device)
    model.eval()
    output = model(data_tensor)
    result = np.round(output.detach().numpy())
    return result

if __name__ == '__main__':
    data, labels = read_imdb_data() # load dataset
    # Combine positive and negative reviews along labels, then shuffle the dataset
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    # print the number of reviews in the dataset
    print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
    # print an example of review before and after the transformation
    print("Before => ", train_X[200], "\nAfter => ", review_to_words(train_X[200]))
    # Convert each review to a list of words
    train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
    word_dict = build_dict(train_X) # create words dictionary
    # export words dictionary to a file
    data_dir = Path('./data/pytorch')
    data_dir.mkdir(exist_ok=True)
    with open(data_dir.joinpath('word_dict.pkl'), "wb") as f:
        pickle.dump(word_dict, f)

    train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
    test_X, test_X_len = convert_and_pad_data(word_dict, test_X)

    # convert dataset to a dataframe
    pd.concat([
        pd.DataFrame(train_y),
        pd.DataFrame(train_X_len),
        pd.DataFrame(train_X)],
        axis=1) \
        .to_csv(data_dir.joinpath('train.csv'), header=False, index=False)

    # Read in only the first 250 rows
    dataset = pd.read_csv(str(data_dir.joinpath('train.csv')), header=None, names=None, nrows=2000)

    train_sample, validation_sample = train_test_split(dataset, test_size=0.3)
    print(train_sample.groupby([0]).size()) # number of elements by class in the training set
    print(validation_sample.groupby([0]).size()) # number of elements by class in the validation set

    # # Turn the input pandas dataframe into tensors
    train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze() # grab only the o columns
    train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long() # remove the first column
    # Build the dataset
    train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
    # Build the dataloader
    train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)
    #
    # # # Turn the input pandas dataframe into tensors
    val_sample_y = torch.from_numpy(validation_sample[[0]].values).float().squeeze()  # grab only the o columns
    val_sample_X = torch.from_numpy(validation_sample.drop([0], axis=1).values).long()  # remove the first column
    # Build the dataset
    val_sample_ds = torch.utils.data.TensorDataset(val_sample_X, val_sample_y)
    # Build the dataloader
    val_sample_dl = torch.utils.data.DataLoader(val_sample_ds, batch_size=50)

    #sample_x, sample_y = next(train_sample_dl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(32, 200, 5000).to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    #train(model, train_sample_dl,val_sample_dl,  10, optimizer, loss_fn, device)

    test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'
    #test_review = "For shockingly long stretches, this new Lego Movie is more of an ungainly, plodding jumble than a functional film."
    #test_review = "Like the plastic bricks themselves, these movies are as much fun for adults as for kids"

    model = LSTMClassifier(32, 200, 5000)
    model.load_state_dict(torch.load("./model_last.pt"))  # it takes the loaded dictionary, not the path file itself
    prediction = predict(model, test_review)
    print(prediction)
