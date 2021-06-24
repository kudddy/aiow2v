import logging
import re

from pymystem3 import Mystem
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.setLevel(logging.CRITICAL)


class Tokenizer:
    def __init__(self):
        self.space_pattern = re.compile(r'[^.А-ЯA-ZЁ]+', re.I)

        self.m = Mystem()

        try:
            with open('nw_model/stopwords.txt') as f:
                self.stop_words = set(f.read().split('\n')) | {''}
        except FileNotFoundError:
            self.stop_words = set()
            logging.critical("WARNING!!! Stop-words file not found!")

    def tokenize_line(self, line):
        """
        Токенизирует одну строку
        :param line:
        :return: набор лексем (pymysteam)
        """
        try:
            return [word for word in self.m.lemmatize(
                self.space_pattern.sub(' ', line.lower())) if word.strip() not in self.stop_words]
        except BrokenPipeError:
            self.m = Mystem()
            return self.tokenize_line(line)

    def join(self, lst):
        return self.space_pattern.sub(' ', ' '.join(lst))


class FastTextVectorizer(Tokenizer):
    def __init__(self, filename):
        super().__init__()
        self.fasttext_model = KeyedVectors.load(filename)

        self.vocab = self.fasttext_model.wv.vocab
        self.wv = self.fasttext_model.wv
        self.vector_size = self.fasttext_model.vector_size

        self.Z = 0
        for k in self.fasttext_model.wv.vocab:
            self.Z += self.fasttext_model.wv.vocab[k].count

    def get_mean_fasttext_vector(self, tokenized_doc):
        # создаем маски для векторов
        lemmas_vectors = np.zeros((len(tokenized_doc), self.fasttext_model.vector_size))

        # если слово есть в модели, берем его вектор
        for idx, lemma in enumerate(tokenized_doc):
            if lemma in self.fasttext_model.vocab:
                lemmas_vectors[idx] = self.fasttext_model.wv[lemma]
        return np.mean(lemmas_vectors, axis=0)

    def fit_transform(self, data):
        array = self.get_sif_vectors(data)
        return array

    def transform(self, data):
        return self.fit_transform(data)

    def get_sif_vectors(self, sents, is_tokenize=True, alpha=1e-3):

        output = []

        for s in sents:
            v = self.get_sif_vector(s, is_tokenize, alpha)
            output.append(v)
        return np.vstack(output)

    def get_sif_vector(self, tokenize_doc, is_tokenize=True, alpha=1e-3):
        count = 0
        v = np.zeros(self.vector_size)
        if is_tokenize:
            tokenize_doc = self.tokenize_line(tokenize_doc)
        for w in tokenize_doc:
            if w in self.vocab:
                v += (alpha / (alpha + (self.vocab[w].count / self.Z))) * self.wv[w]
                count += 1

        if count > 0:
            for i in range(self.vector_size):
                v[i] *= 1 / count
        return v
