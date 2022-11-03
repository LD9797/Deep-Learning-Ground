from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocesar_documento_1(document):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)
    word_tokens = [token.lower() for token in word_tokens]
    filtration = [word for word in word_tokens if word not in stop_words]
    return filtration


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


def preprocesar_documento_2(document):
    word_tokens = word_tokenize(document)
    lemmatizer = WordNetLemmatizer()
    filtration = [lemmatizer.lemmatize(word) for word in word_tokens]
    return filtration

