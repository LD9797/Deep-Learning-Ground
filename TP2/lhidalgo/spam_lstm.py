# https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn
# https://github.com/FernandoLpz/Text-Classification-LSTMs-PyTorch
# https://towardsdatascience.com/text-classification-with-pytorch-7111dae111a6
import pandas as pd
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd



def preprocesar_documento_1(document, to_string=False):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)
    word_tokens = [token.lower() for token in word_tokens]
    word_tokens = [token for token in word_tokens if token.isalnum()]  # To remove punctuations
    filtration = [word for word in word_tokens if word not in stop_words]
    return filtration if not to_string else ' '.join(filtration)


def preprocesar_documento_2(document, to_string=False):
    word_tokens = word_tokenize(document)
    lemmatizer = WordNetLemmatizer()
    word_tokens = [token.lower() for token in word_tokens]
    word_tokens = [token for token in word_tokens if token.isalnum()]  # To remove punctuations
    filtration = [lemmatizer.lemmatize(word, pos="v") for word in word_tokens]
    return filtration if not to_string else ' '.join(filtration)


class Preprocessing:

    def __init__(self):
        """
        Class constructor
        """
        self.data = 'natural_disaster.csv'
        # maximum length for each sequence, CORRECT
        self.max_len = 200
        # Maximum number of words in the dictionary
        self.max_words = 200
        # percentage of test data
        self.test_size = 0.2

    def load_data(self):
        """
        Loads and splits the data
        """
        # load training and test data
        df = pd.read_csv(self.data)
        # eliminate unnecesary information from training data
        df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)
        # extract input and labels
        X = df['text'].values
        Y = df['target'].values
        # create train/test split using sklearn
        # HAM 0
        # SPAM 1
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size)

    def prepare_tokens(self):
        """
        Tokenizes the input text
        """
        # tokenize the input text
        self.tokens = Tokenizer(num_words=self.max_words)
        self.tokens.fit_on_texts(self.x_train)

    def sequence_to_token(self, x):
        """
        Converts the input sequence of strings to a sequence of integers
        """
        # transform the token list to a sequence of integers
        sequences = self.tokens.texts_to_sequences(x)
        # add padding using the maximum length specified
        # Rellena con 0 hacia la izquierda y deja la palabra en una lista de len self.max_len
        return keras.utils.pad_sequences(sequences, maxlen=self.max_len)




class LSTM_TweetClassifier(nn.ModuleList):

    def __init__(self, batch_size=64, hidden_dim=20, lstm_layers=2, max_words=200):
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
        # print("x shape ", x.shape)
        # print("embedding ", self.embedding)
        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)

        #  Fully connected network para la clasificacion
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        # sigmoid activation function
        out = torch.sigmoid(self.fc2(out))

        return out


class DatasetMaper(Dataset):
    '''
    Handles batches of dataset
    '''

    def __init__(self, x, y):
        """
        Inits the dataset mapper
        """
        self.x = x
        self.y = y

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Fetches a specific item by id
        """
        return self.x[idx], self.y[idx]


def create_data_loaders(batch_size = 64):
  preprocessor = Preprocessing()
  #load the data
  preprocessor.load_data()
  #tokenize the text
  preprocessor.prepare_tokens()
  raw_x_train = preprocessor.x_train
  raw_x_test = preprocessor.x_test
  y_train = preprocessor.y_train
  y_test = preprocessor.y_test
  #convert sequence of strings to tokens
  x_train = preprocessor.sequence_to_token(raw_x_train)
  x_test = preprocessor.sequence_to_token(raw_x_test)
  #create data loaders
  training_set = DatasetMaper(x_train, y_train)
  test_set = DatasetMaper(x_test, y_test)
  loader_training = DataLoader(training_set, batch_size=batch_size)
  loader_test = DataLoader(test_set)
  return loader_training, loader_test


loader_training, loader_test = create_data_loaders()