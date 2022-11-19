from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np


COLLECTION_PATH = ".\\smsspamcollection"
MAX_WORDS = 200
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_DIM = 20
LSTM_LAYERS = 2
TEST_SIZE = 0.4



# Processing
def preprocesar_documento_1(document, to_string=False):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)
    word_tokens = [token.lower() for token in word_tokens]
    word_tokens = [token for token in word_tokens if token.isalnum()]  # To remove punctuations
    filtration = [word for word in word_tokens if word not in stop_words]
    return filtration if not to_string else ' '.join(filtration)


def preprocesar_documento_2(document, to_string=False):
    word_tokens = preprocesar_documento_1(document)
    lemmatizer = WordNetLemmatizer()
    filtration = [lemmatizer.lemmatize(word, pos="v") for word in word_tokens]
    return filtration if not to_string else ' '.join(filtration)


#  Function Equivalent to prepare_tokens
def tokens_to_indexes(sentences: list) -> dict:
    words_count = {}
    for sentence in sentences:
        for word in sentence:
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1
    words_to_list = list(dict(sorted(words_count.items(), key=lambda item: item[1])))
    words_to_list.reverse()
    top_max_words = words_to_list[0: MAX_WORDS - 1]
    index_words = {}
    counter = 1
    for word in top_max_words:
        index_words[word] = counter
        counter += 1
    return index_words


def sequence_to_number_combination(word: list, index_words: dict):
    sequence = []
    for token in word:
        if token in index_words:
            sequence.append(index_words[token])
    return sequence


def adapt_to_input_layer(dataset, input_layer_size=MAX_WORDS):
    new_dataset = []
    for data in dataset:
        zeros_to_add = input_layer_size - len(data)
        new_data_list = [0 for zero in range(zeros_to_add)]
        new_data_list.extend(data)
        new_dataset.append(new_data_list)
    return new_dataset


# Data iterator
class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Model !
class LSTM_TweetClassifier(nn.ModuleList):

    def __init__(self, batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, lstm_layers=LSTM_LAYERS, max_words=MAX_WORDS):
        """
        param batch_size: batch size for training data
        param hidden_dim: number of hidden units used in the LSTM and the Embedding layer
        param lstm_layers: number of lstm_layers
        param max_words: maximum sentence length
        """
        super(LSTM_TweetClassifier, self).__init__()
        # batch size during training
        self.batch_size = batch_size
        # number of hidden units in the LSTM layer
        self.hidden_dim = hidden_dim
        # Number of LSTM layers
        self.LSTM_layers = lstm_layers
        self.input_size = max_words  # embedding dimension

        self.dropout = nn.Dropout(0.5)  # Para descartar
        #  N, D			#  hidden_dim -> Determina el tamaño del embedding
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)  # Aprender la representacion
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)  # Capaz de aprender/olvidar dependiendo de las relaciones.
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=257)
        self.fc2 = nn.Linear(257, 1)

    def forward(self, x):
        """
        Forward pass
        param x: model input
        """
        # it starts with noisy estimations of h and c
        #  Context y estado
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))  # "Contexto"
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))  # "Estado"
        # Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
        # The resulting tensor will have values sampled from \mathcal{N}(0, \text{std}^2)N(0,std)
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)

        #  Fully connected network para la clasificacion
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        # sigmoid activation function
        out = torch.sigmoid(self.fc2(out))

        return out


#  Data loader
def load_process_data():
    batch_size = BATCH_SIZE
    # Load dataset
    dataset_frame = pd.read_csv('smsspamcollection\\SMSSpamCollection', delimiter='\t', header=None)
    # Preprocess document
    sentences_list = [preprocesar_documento_1(sentence) for sentence in dataset_frame[1]]
    # Add index to every word
    words_dictionary = tokens_to_indexes(sentences_list)
    # Transform tokens (words) to indexes (numbers)
    sentences_list = [sequence_to_number_combination(sentence, words_dictionary) for sentence in sentences_list]
    # One-hot tags ham = 0, spam = 1
    tags_list = [0 if tag == "ham" else 1 for tag in dataset_frame[0]]
    # Build train and test datasets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(sentences_list, tags_list, test_size=0.4)
    # Adapt data to input layer
    x_train = adapt_to_input_layer(X_train_raw)
    x_test = adapt_to_input_layer(X_test_raw)
    training_set = DatasetMaper(x_train, y_train)
    test_set = DatasetMaper(x_test, y_test)
    loader_training = DataLoader(training_set, batch_size=batch_size)
    loader_test = DataLoader(test_set)
    return loader_training, loader_test


# Model evaluation
def calculate_accuray(y_pred, y_gt):
    return accuracy_score(y_pred, y_gt)


def evaluate_model(model, loader_test):
    predictions = []
    accuracies = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            x_batch = torch.t(torch.stack(x_batch))
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)
            y_pred = model(x)
            y_pred = torch.round(y_pred).flatten()
            predictions += list(y_pred.detach().numpy())
            acc_batch = accuracy_score(y_pred, y)
            accuracies += [acc_batch]
    return np.array(accuracies)


def train_model(model, epochs=10, learning_rate=0.01):
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        predictions = []
        model.train()
        loss_dataset = 0
        for x_batch, y_batch in loader_training:
            x_batch = torch.t(torch.stack(x_batch))
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)
            try:
                y_pred = model(x).flatten()
                loss = F.binary_cross_entropy(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_dataset += loss
            except Exception:
                pass
        accuracies = evaluate_model(model, loader_test)
        print("Epoch ", epoch, " Loss training : ", loss_dataset.item(), " Accuracy test: ", accuracies.mean())



loader_training, loader_test = load_process_data()
model = LSTM_TweetClassifier()
train_model(model, EPOCHS, LEARNING_RATE)
accuracies = evaluate_model(model, loader_test)
print("average accuracy : ", accuracies.mean())
