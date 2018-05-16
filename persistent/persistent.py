#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Alexey Koshevoy

import dionysus as dn
import numpy as np
from gensim.models import word2vec
import pymorphy2
import nltk
from string import punctuation
import random


PATH = '/Users/alexey/Documents/GitHub/dialog_persistent/stop words/sw.txt'

with open(PATH) as f:
    stop_words = f.read().splitlines()

morph = pymorphy2.MorphAnalyzer()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class PersistentPreTrained:

    def __init__(self, model,
                 path=None,
                 text=None,
                 split_sent=None,
                 check=False):

        self.text = text
        self.path = path

        if split_sent:
            self.tokenized = split_sent
        else:
            self.tokenized = self._tokenize()

        self.check = check
        self.model = model

    @staticmethod
    def check_sentence(sentence):

        """
        This static method is used to check the sentence for stop words
        :param sentence: list of words from _tokenize
        :return: sentence cleared from stop-words
        """
        for element in sentence:
            if element in stop_words:
                sentence.remove(element)

        return sentence

    def _tokenize(self):

        """
        This private method is used to tokenize given text and split it into
        lists of words, contained in one sentence

        :return: list o lists, which contains sentences
        """
        if self.path:
            file = open(self.path).read() 
        else:
            file = self.text

        sentences = tokenizer.tokenize(file)

        split_sent = []

        for s in sentences:
            s = s.translate(str.maketrans('', '', punctuation))
            split_sent.append(s.lower().split())

        for sentence in split_sent:
            for i in range(len(sentence)):
                p = morph.parse(sentence[i])[0]
                sentence[i] = p.normal_form

        if self.check:
            for i in range(len(split_sent)):
                split_sent[i] = self.check_sentence(split_sent[i])

        return split_sent

    def _get_vectors(self):

        vocab = self.tokenized

        tokens = []

        for word in vocab:
            try:
                tokens.append(self.model[word])
            except KeyError:
                pass

        result_array = np.empty((0, 300))

        for token in tokens:
            result_array = np.append(result_array, [token], axis=0)

        return result_array
        # return tokens

    def persistent(self):

        """
        This method is used to create dionysus persistent homology
        implementation of given vectors
        :return: dionysus diagrams
        """
        vectors = self._get_vectors()

        f_lower_star = dn.fill_freudenthal(vectors)
        p = dn.homology_persistence(f_lower_star)
        dgms = dn.init_diagrams(p, f_lower_star)

        return dgms

    def __str__(self):
        return self.tokenized


class PersistentNonTrained:

    def __init__(self, path=None, text=None, size=100, window=20, min_count=2,
                 workers=4, shuffle=False, split_sent=None):

        self.shuffle = shuffle
        self.text = text
        self.path = path

        if split_sent:
            self.tokenized = split_sent
        else:
            self.tokenized = self._tokenize()

        # model parameters

        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        self.model = self._create_w2v()

    @staticmethod
    def check_sentence(sentence):

        """
        This static method is used to check the sentence for stop words
        :param sentence: list of words from _tokenize
        :return: sentence cleared from stop-words
        """
        for element in sentence:
            if element in stop_words:
                sentence.remove(element)

        return sentence

    def _tokenize(self):

        """
        This private method is used to tokenize given text and split it into
        lists of words, contained in one sentence
        :return: list o lists, which contains sentences
        """
        if self.path:
            file = open(self.path).read()
        else:
            file = self.text

        sentences = tokenizer.tokenize(file)

        split_sent = []

        for s in sentences:
            s = s.translate(str.maketrans('', '', punctuation))
            split_sent.append(s.lower().split())

        if self.shuffle:
            split_sent = random.shuffle(split_sent, random.random)

        for sentence in split_sent:
            for i in range(len(sentence)):
                p = morph.parse(sentence[i])[0]
                sentence[i] = p.normal_form

        for i in range(len(split_sent)):
            split_sent[i] = self.check_sentence(split_sent[i])

        return split_sent

    def _create_w2v(self):

        """
        This method is used to create w2v model with given hyper-parameters
        :return: w2v model
        """
        model = word2vec.Word2Vec(self.tokenized,
                                  size=self.size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  workers=self.workers)

        return model

    def _get_vectors(self):

        tokens = []

        for word in self.model.wv.vocab:
            tokens.append(self.model[word])

        result_array = np.empty((0, self.size))

        for token in tokens:
            result_array = np.append(result_array, [token], axis=0)

        return result_array

    def persistent(self):

        """
        This method is used to create dionysus persistent homology
        implementation of given vectors
        :return: dionysus diagrams
        """
        vectors = self._get_vectors()

        f_lower_star = dn.fill_freudenthal(vectors)
        p = dn.homology_persistence(f_lower_star)
        dgms = dn.init_diagrams(p, f_lower_star)

        return dgms

    def __str__(self):
        return self.tokenized
