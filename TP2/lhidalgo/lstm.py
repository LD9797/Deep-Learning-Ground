from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


COLLECTION_PATH = ".\\smsspamcollection"


def preprocesar_documento_1(document, to_string=False):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)
    word_tokens = [token.lower() for token in word_tokens]
    word_tokens = [token for token in word_tokens if token.isalnum()]  # To remove punctuations
    filtration = [word for word in word_tokens if word not in stop_words]
    return filtration if not to_string else ' '.join(filtration)


"""Lemmatize `word` using WordNet's built-in morphy function.
        Returns the input word unchanged if it cannot be found in WordNet.

        :param word: The input word to lemmatize.
        :type word: str
        :param pos: The Part Of Speech tag. Valid options are `"n"` for nouns,
            `"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"`
            for satellite adjectives.
        :param pos: str
        :return: The lemma of `word`, for the given `pos`.
"""


def preprocesar_documento_2(document, to_string=False):
    word_tokens = word_tokenize(document)
    lemmatizer = WordNetLemmatizer()
    word_tokens = [token.lower() for token in word_tokens]
    word_tokens = [token for token in word_tokens if token.isalnum()]  # To remove punctuations
    filtration = [lemmatizer.lemmatize(word, pos="v") for word in word_tokens]
    return filtration if not to_string else ' '.join(filtration)


def preprocess_example():
    with open(COLLECTION_PATH + "\\SMSSpamCollection", 'r') as collection:
        for line in collection:
            line = line.replace("ham", "")
            line = line.replace("spam", "")
            print("Line:")
            print(line)
            print("Preprocess #1")
            print(preprocesar_documento_1(line, to_string=True))
            print("Preprocess #2")
            print(preprocesar_documento_2(line, to_string=True))
            break


#  Function Equivalent to prepare_tokens
def tokens_to_indexes(words: list) -> dict:
    index_words = {}
    counter = 1
    for word in words:
        if word not in index_words:
            index_words[word] = counter
            counter += 1
    return index_words


def sequence_to_number_combination(word: list, index_words: dict):
    return [index_words[token] for token in word]


#spam = "Free Free entry in 2 a wkly entry comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
#spam = preprocesar_documento_2(spam)
#words_indexes = tokens_to_indexes(spam)
#spam_to_numbers = sequence_to_number_combination(spam, words_indexes)
#pass


#  Code equivalent to load_data
dataset_frame = pd.read_csv('smsspamcollection\\SMSSpamCollection', delimiter='\t', header=None)
#  To normal lists
sentences_list = [sentence for sentence in dataset_frame[1]]
tags_list = [0 if tag == "ham" else 1 for tag in dataset_frame[0]]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(sentences_list, tags_list, test_size=0.4)
pass

# TODO replicate keras.utils.pad_sequences(sequences, maxlen=self.max_len)



