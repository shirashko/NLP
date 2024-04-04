import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt
from data_loader import get_negated_polarity_examples, get_rare_words_examples, SentimentTreeBank

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

# Constants we defined for the models
TRAINING_BATCH_SIZE = 64

LOG_LINEAR_LEARNING_RATE = 0.01
LOG_LINEAR_WEIGHT_DECAY = 0.001
LOG_LINEAR_EPOCHS = 20

LSTM_SENTENCE_LENGTH = 52
LSTM_LEARNING_RATE = 0.001
LSTM_WEIGHT_DECAY = 0.0001
LSTM_DROP_OUT = 0.5
LSTM_EPOCHS = 4

LOG_LINEAR_ONE_HOT_MODEL_PATH = "log_linear_one_hot_model.pth"
LOG_LINEAR_W2V_MODEL_DIR = "log_linear_w2v_model_training_and_testing_results"
LOG_LINEAR_W2V_MODEL_PATH = "log_linear_w2v_model.pth"
LSTM_MODEL_PATH = "lstm_model.pth"

LOG_LINEAR_ONE_HOT_MODEL_NAME = "Log Linear Model with One Hot Encoding"
LOG_LINEAR_W2V_MODEL_NAME = "Log Linear Model with Word2Vec Average Encoding"
LSTM_MODEL_NAME = "LSTM Model"

N_LAYERS_LSTM_MODEL = 1  # we are using a single layer LSTM model
HIDDEN_DIM_FOR_LSTM_MODEL = 100  # the hidden dimension of the LSTM model


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    """
    saves a python object into a pickle file
    :param obj: the object to save
    :param path: the path to save the object into
    :return: None
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
     loads a python object from a pickle file
    :param path: the path to load the object from
    :return: the loaded object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param path: path to save the checkpoint into
    :param epoch: the current epoch
    :param optimizer: torch optimizer used for training the module
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each length 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    sentence_words = sent.text  # list of words of the sentence
    word_embeddings = []
    for word in sentence_words:
        if word in word_to_vec:  # Average without the unknowns
            word_embeddings.append(word_to_vec[word])
    if len(word_embeddings) == 0:
        return np.zeros(embedding_dim)  # if all words are unknown return zeros
    return np.mean(np.array(word_embeddings), axis=0)  # return the average of the known words


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot_vector = np.zeros(size)
    one_hot_vector[ind] = 1
    return one_hot_vector


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return: numpy ndarray of shape (vocabulary_size,) with the average one-hot embedding of the sentence
    """
    sentence_words = sent.text
    vocabulary_length, sentence_length = len(word_to_ind), len(sentence_words)
    one_hot_vectors = np.array([get_one_hot(vocabulary_length, word_to_ind[word]) for word in sentence_words])
    return np.mean(one_hot_vectors, axis=0)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: i for i, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    sentence_words = sent.text  # list of words of the sentence
    sentence_length = len(sentence_words)
    sentence_embedding = np.zeros((seq_len, embedding_dim))  # create a zero array to host the sentence embeddings
    for i in range(min(seq_len, sentence_length)):  # copy the embeddings to the array, no more than seq_len
        word = sentence_words[i]  # get the word
        if word in word_to_vec:  # if the word is known use it's embedding
            sentence_embedding[i] = word_to_vec[word]
        # else use zero which we already initialized the array with
    return sentence_embedding


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preparation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the dataset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the dataset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        """
        Initializes the LSTM model.
        :param embedding_dim: Dimension of the Word2Vec embeddings.
        :param hidden_dim: Dimension of the hidden and cell states of the LSTM.
        :param n_layers: Number of layers in the LSTM.
        :param dropout: Dropout rate for the dropout layer.
        """
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,  # Changes the expected input shape to have the batch size as the first
                            # dimension instead of the sequence length. Adjusts the output shape
                            # accordingly, so the batch size also comes first.
                            dropout=dropout if n_layers > 1 else 0,  # When the LSTM has only one layer, there are no
                            # "interlayer" connections where dropout could be applied
                            bidirectional=True)

        # Linear layer to bring back to 1 dimension output
        self.linear = nn.Linear(hidden_dim * 2, 1)  # Times 2 because it's bidirectional

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """
        Forward pass through the network.
        :param text: Input tensor of shape (batch_size, seq_len, embedding_dim).
        :return The output of the network.
        """
        # Pass the input through the LSTM layer
        lstm_output, (hidden_state, cell_state) = self.lstm(text.float())

        # Concatenate the final forward and backward hidden states and pass them through the dropout layer
        lstm_out = self.dropout(torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1))

        # Pass the output through the linear layer
        output = self.linear(lstm_out)

        return output.squeeze()

    def predict(self, text):
        """
        Predict the sentiment of the text.
        :param text: Input tensor of shape (batch_size, seq_len, embedding_dim).
        :return The prediction of the network.
        """
        with torch.no_grad():
            outputs = self.forward(text)
            prediction = torch.sigmoid(outputs)  # Apply the non-linear sigmoid function to the output
        return prediction.squeeze()


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        """
        :param embedding_dim: the dimension of the word embeddings used.
        """
        super().__init__()
        self.linear_layer = nn.Linear(embedding_dim, 1, bias=True)  # Linear layer
        self.output_activation = nn.Sigmoid()  # non-linear activation function

    def forward(self, x):
        """
        pass the input through the network
        :param x: the input tensor, of shape (batch_size, embedding_dim)
        :return: the output of the network
        """
        x = x.to(self.linear_layer.weight.dtype)  # Ensure the input tensor has the same dtype as the weights
        return self.linear_layer(x).squeeze()

    def predict(self, x):
        """
        creates a prediction for the input x
        :param x: the input tensor, of shape (batch_size, embedding_dim)
        :return: the prediction of the network
        """
        with torch.no_grad():  # No need to track the gradients
            return self.output_activation(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(predictions, true_labels):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param predictions: a vector of float predictions
    :param true_labels: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    accurate_predictions = 0
    num_predictions = len(predictions)
    for i in range(num_predictions):
        if torch.round(predictions[i]) == true_labels[i]:
            accurate_predictions += 1
    return accurate_predictions / num_predictions


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    :return the accuracy and loss for this epoch
    """
    # set the model to training mode and go over the training data, and update the model's weights
    model.train()
    for batch_samples, batch_true_labels in data_iterator:
        optimizer.zero_grad()
        outputs = model(batch_samples)  # Automatically call forward
        loss = criterion(outputs, batch_true_labels)
        loss.backward()
        optimizer.step()

    # At the end of updating the model's weights, we return the accuracy and loss for this epoch
    return evaluate(model, data_iterator, criterion)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models.
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    loss, accuracy = 0, 0
    with torch.no_grad():
        for batch_samples, batch_labels in data_iterator:
            outputs = model(batch_samples)
            loss += criterion(outputs, batch_labels).item()
            accuracy += binary_accuracy(model.predict(batch_samples), batch_labels)
    number_of_batches = len(data_iterator)
    return loss / number_of_batches, accuracy / number_of_batches


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return: numpy ndarray or torch tensor with the model predictions
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_samples, _ in data_iter:
            prediction = model.predict(batch_samples)
            predictions.append(prediction)
    return torch.cat(predictions)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    :return: a tuple of lists. (train_losses, train_accuracies, validation_losses, validation_accuracies)
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize lists to store losses and accuracies for each epoch
    train_epochs_loss, train_epochs_accuracy = [], []
    validation_epochs_loss, validation_epochs_accuracy = [], []

    for _ in tqdm.tqdm(range(n_epochs)):
        train_epoch_loss, train_epoch_accuracy = train_epoch(model, data_manager.get_torch_iterator(TRAIN), optimizer,
                                                             criterion)
        train_epochs_loss.append(train_epoch_loss), train_epochs_accuracy.append(train_epoch_accuracy)

        val_epoch_loss, val_epoch_accuracy = evaluate(model, data_manager.get_torch_iterator(VAL), criterion)
        validation_epochs_loss.append(val_epoch_loss), validation_epochs_accuracy.append(val_epoch_accuracy)

    return train_epochs_loss, train_epochs_accuracy, validation_epochs_loss, validation_epochs_accuracy


def train_log_linear_with_one_hot():
    """
    Train the log linear model with one-hot encoding representation.
    :return: the trained model and the data manager used for training
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=TRAINING_BATCH_SIZE)
    vocab_size = data_manager.get_input_shape()[0]  # Size of the vocabulary == num of features in the one-hot encoding
    model = LogLinear(embedding_dim=vocab_size).to(get_available_device())
    train_epochs_loss, train_epochs_accuracy, validation_epochs_loss, validation_epochs_accuracy = (
        train_model(model, data_manager, n_epochs=LOG_LINEAR_EPOCHS, lr=LOG_LINEAR_LEARNING_RATE,
                    weight_decay=LOG_LINEAR_WEIGHT_DECAY))

    save_model(model, LOG_LINEAR_ONE_HOT_MODEL_PATH, LOG_LINEAR_EPOCHS,
               optim.Adam(model.parameters(), lr=LOG_LINEAR_LEARNING_RATE, weight_decay=LOG_LINEAR_WEIGHT_DECAY))

    save_and_plot_results(train_epochs_accuracy, train_epochs_loss, validation_epochs_accuracy,
                          validation_epochs_loss, LOG_LINEAR_ONE_HOT_MODEL_NAME, LOG_LINEAR_EPOCHS)

    return data_manager, model


def train_log_linear_with_w2v():
    """
    Train the log linear model with Word2Vec average encoding representation.
    :return: the trained model and the data manager used for training
    """
    # Create all the objects needed for the training process,
    data_manager = DataManager(batch_size=TRAINING_BATCH_SIZE, data_type=W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM)
    vocab_size = data_manager.get_input_shape()[0]
    model = LogLinear(vocab_size).to(get_available_device())
    training_epochs_loss, training_epochs_accuracy, validation_epochs_loss, validation_epochs_accuracy = (
        train_model(model, data_manager, n_epochs=LOG_LINEAR_EPOCHS, lr=LOG_LINEAR_LEARNING_RATE,
                    weight_decay=LOG_LINEAR_WEIGHT_DECAY))

    save_model(model, LOG_LINEAR_W2V_MODEL_PATH, LOG_LINEAR_EPOCHS,
               optim.Adam(model.parameters(), lr=LOG_LINEAR_LEARNING_RATE, weight_decay=LOG_LINEAR_WEIGHT_DECAY))

    save_and_plot_results(training_epochs_accuracy, training_epochs_loss, validation_epochs_accuracy,
                          validation_epochs_loss, LOG_LINEAR_W2V_MODEL_NAME, LOG_LINEAR_EPOCHS)

    return data_manager, model


def train_lstm_with_w2v():
    """
    Train the LSTM model with Word2Vec embeddings.
    :return: the trained model and the data manager used for training with the necessary configuration.
    """
    # First, initialize the DataManager to provide data in the W2V_SEQUENCE format
    data_manager = DataManager(data_type=W2V_SEQUENCE,
                               batch_size=TRAINING_BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)

    # Initialize the LSTM model
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM_FOR_LSTM_MODEL,  # Example hidden_dim, adjust based on experiment
                 n_layers=N_LAYERS_LSTM_MODEL,  # Example n_layers, adjust based on experiment
                 dropout=LSTM_DROP_OUT).to(get_available_device())

    # Train the model
    train_epochs_loss, train_epochs_accuracy, validation_epochs_loss, validation_epochs_accuracy = (
        train_model(model, data_manager, n_epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE, weight_decay=LSTM_WEIGHT_DECAY))

    # Save the trained model
    save_model(model, LSTM_MODEL_PATH, LSTM_EPOCHS, optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE,
                                                               weight_decay=LSTM_WEIGHT_DECAY))

    # Optionally, save and plot the training results
    save_and_plot_results(train_epochs_accuracy, train_epochs_loss, validation_epochs_accuracy, validation_epochs_loss,
                          LSTM_MODEL_NAME, LSTM_EPOCHS)

    return data_manager, model


# ------------------------- testing functions -------------

def test_lstm_with_w2v(data_manager, trained_model):
    """
    Test the LSTM model with Word2Vec embeddings on the test dataset.

    :param trained_model: The trained LSTM model.
    :param data_manager: DataManager object with the necessary configuration to provide test data.
    """
    # Get the test data iterator
    test_iterator = data_manager.get_torch_iterator(TEST)
    # Evaluate the model on the test set
    test_loss, test_accuracy = evaluate(trained_model, test_iterator, nn.BCEWithLogitsLoss().to(get_available_device()))
    # Evaluate the model on the special cases
    negated_polarity_accuracy, rare_words_accuracy = test_model_special_cases(trained_model, data_manager)

    save_testing_results(test_loss, test_accuracy, negated_polarity_accuracy, rare_words_accuracy, LSTM_MODEL_NAME)


def test_model_special_cases(model, data_manager):
    """
    Test the model on the groups of special sentences, which are:
    1. negated polarity sentences
    2. sentences with rare words
    :param model: one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :return: The accuracy of the model on the special sentences
    """
    test_set = data_manager.sentences[TEST]

    # Negated polarity sentences
    negated_polarity_indices = get_negated_polarity_examples(test_set)
    negated_polarity_sentences = [test_set[i] for i in negated_polarity_indices]
    negated_polarity_accuracy = test_special_case(model, data_manager, negated_polarity_sentences)

    # Sentences with rare words
    rare_words_indices = get_rare_words_examples(test_set, data_manager.sentiment_dataset)
    rare_words_sentences = [test_set[i] for i in rare_words_indices]
    rare_words_accuracy = test_special_case(model, data_manager, rare_words_sentences)

    return negated_polarity_accuracy, rare_words_accuracy


def test_model_on_test_set(model, data_manager):
    """
    Test the model on the test set using the DataManager object
    :param model: one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :return: a tuple of (test_loss, test_accuracy)
    """
    test_iterator = data_manager.get_torch_iterator(TEST)
    return evaluate(model, test_iterator, nn.BCEWithLogitsLoss())


def test_special_case(model, data_manager, sentences):
    """
    Test the model on a group of special sentences
    :param model: the model to be tested
    :param data_manager: the DataManager object
    :param sentences: the special sentences to be tested
    :return: the accuracy of the model on the special sentences
    """
    dataset = OnlineDataset(sentences, data_manager.sent_func, data_manager.sent_func_kwargs)
    data_iterator = DataLoader(dataset, batch_size=TRAINING_BATCH_SIZE)  # More efficient batch size
    _, accuracy = evaluate(model, data_iterator, nn.BCEWithLogitsLoss())
    return accuracy


# ------------------------- utility functions -------------

def save_and_plot_results(training_epochs_accuracy, training_epochs_loss, validation_epochs_accuracy,
                          validation_epochs_loss, model_name, number_of_epochs):
    """
    saving the training results to a file and plotting the training and validation loss and accuracy.
    :param training_epochs_accuracy: a list of the training accuracy of each epoch
    :param training_epochs_loss: a list of the training loss of each epoch
    :param validation_epochs_accuracy: a list of the validation accuracy of each epoch
    :param validation_epochs_loss: a list of the validation loss of each epoch
    :param model_name: the name of the model
    :param number_of_epochs: the number of epochs
    """
    save_training_results(training_epochs_loss, training_epochs_accuracy, validation_epochs_loss,
                          validation_epochs_accuracy, model_name, number_of_epochs)
    plot_training_and_validation(training_epochs_loss, training_epochs_accuracy, validation_epochs_loss,
                                 validation_epochs_accuracy, model_name, number_of_epochs)


def test_log_linear_model(data_manager, trained_model, model_name):
    """
    Test the log linear model on the test dataset.
    :param trained_model: the trained log linear model
    :param data_manager: the DataManager object to provide test data
    :param model_name: the name of the model
    """
    test_loss, test_accuracy = test_model_on_test_set(trained_model, data_manager)
    negated_polarity_accuracy, rare_words_accuracy = test_model_special_cases(trained_model, data_manager)
    save_testing_results(test_loss, test_accuracy, negated_polarity_accuracy, rare_words_accuracy, model_name)


def save_training_results(training_epochs_loss, training_epochs_accuracy, validation_epochs_loss,
                          validation_epochs_accuracy, model_name, number_of_epochs):
    """
    This method saves the training results to a file.
    :param training_epochs_loss: the training loss of each epoch
    :param training_epochs_accuracy: the training accuracy of each epoch
    :param validation_epochs_loss: the validation loss of each epoch
    :param validation_epochs_accuracy: the validation accuracy of each epoch
    :param model_name: the name of the model
    :param number_of_epochs: the number of epochs the model was trained for
    """
    with open(model_name.replace(" ", "_") + "_training_results.txt", "w") as results_file:
        results_file.write("######################## Training " + model_name + " ########################\n\n")
        for epoch in range(number_of_epochs):
            results_file.write(f"############### Epoch {epoch + 1} ###############\n")
            results_file.write(f"Training Loss: {training_epochs_loss[epoch]}\n")
            results_file.write(f"Training Accuracy: {training_epochs_accuracy[epoch]}\n")
            results_file.write(f"Validation Loss: {validation_epochs_loss[epoch]}\n")
            results_file.write(f"Validation Accuracy: {validation_epochs_accuracy[epoch]}\n")
            results_file.write(f"############### Epoch {epoch + 1} ###############\n\n")
        results_file.write("######################## Training " + model_name + " ########################\n\n")


def save_testing_results(test_loss, test_accuracy, negated_polarity_accuracy, rare_words_accuracy, model_name):
    """
    This method saves the testing results to a file.
    :param test_loss: the loss of the model on the test set
    :param test_accuracy: the accuracy of the model on the test set
    :param negated_polarity_accuracy: the accuracy of the model on the negated polarity sentences
    :param rare_words_accuracy: the accuracy of the model on the sentences with rare words
    :param model_name: the name of the model
    """
    with open(model_name.replace(" ", "_") + "_testing_results.txt", "w") as results_file:
        results_file.write("######################## Testing " + model_name + " ########################\n\n")
        results_file.write(f"Test Loss: {test_loss}\n")
        results_file.write(f"Test Accuracy: {test_accuracy}\n")
        results_file.write(f"Negated Polarity Accuracy: {negated_polarity_accuracy}\n")
        results_file.write(f"Rare Words Accuracy: {rare_words_accuracy}\n\n")
        results_file.write("######################## Testing " + model_name + " ########################\n\n")


def plot_training_and_validation(training_epochs_loss, training_epochs_accuracy,
                                 validation_epochs_loss, validation_epochs_accuracy,
                                 model_name, number_of_epochs):
    """
    Plots the training and validation loss and accuracy of the models.
    :param training_epochs_loss: list of the training loss of each epoch
    :param training_epochs_accuracy: list of the accuracy of each epoch
    :param validation_epochs_loss: list of the  validation loss of each epoch
    :param validation_epochs_accuracy: list of the validation accuracy of each epoch
    :param model_name: the name of the model to be in the title of the plots
    :param number_of_epochs: the number of epochs
    """
    # plot the training and validation loss
    plt.plot(np.arange(1, number_of_epochs + 1), training_epochs_loss, label='Training loss')
    plt.plot(np.arange(1, number_of_epochs + 1), validation_epochs_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, number_of_epochs + 1))
    plt.ylabel('Loss')
    plt.title('Training and validation loss of the ' + model_name,
              fontsize=9, loc='center')
    plt.legend()
    plt.savefig(model_name.replace(" ", "_") + '_training_loss.png', dpi=3000)
    plt.show()
    # plot the training and validation accuracy
    plt.plot(np.arange(1, number_of_epochs + 1), training_epochs_accuracy, label='Training accuracy')
    plt.plot(np.arange(1, number_of_epochs + 1), validation_epochs_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, number_of_epochs + 1))
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy of the ' + model_name,
              fontsize=9, loc='center')
    plt.legend()
    plt.savefig(model_name.replace(" ", "_") + '_training_accuracy.png', dpi=3000)
    plt.show()


def print_heading(message):
    """
    This method prints a heading message to the console.
    :param message: the message to be printed
    """
    print("\n" + "#" * 10 + f" {message} " + "#" * 10 + "\n")


# ------------------------- main -------------

if __name__ == '__main__':
    # Train the log linear model with one-hot encoding representation
    print_heading("Start Training the Log Linear Model with One Hot Encoding")
    one_hot_data_manager, ont_hot_log_linear_trained_model = train_log_linear_with_one_hot()

    # Test the log linear model with one-hot encoding representation
    print_heading("Start Testing the Log Linear Model with One Hot Encoding")
    test_log_linear_model(one_hot_data_manager, ont_hot_log_linear_trained_model, LOG_LINEAR_ONE_HOT_MODEL_NAME)

    # Train the log linear model with Word2Vec average encoding representation
    print_heading("Start Training the Log Linear Model with Word2Vec Average Encoding")
    w2v_data_manager, w2v_log_linear_trained_model = train_log_linear_with_w2v()

    # Test the log linear model with Word2Vec average encoding representation
    print_heading("Start Testing the Log Linear Model with Word2Vec Average Encoding")
    test_log_linear_model(w2v_data_manager, w2v_log_linear_trained_model, LOG_LINEAR_W2V_MODEL_NAME)

    # Train the LSTM model with Word2Vec embeddings
    print_heading("Start Training the LSTM Model")
    lstm_data_manager, lstm_trained_model = train_lstm_with_w2v()

    # Test the LSTM model with Word2Vec embeddings
    print_heading("Start testing the LSTM Model")
    test_lstm_with_w2v(lstm_data_manager, lstm_trained_model)
