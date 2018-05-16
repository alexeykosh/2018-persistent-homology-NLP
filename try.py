#!/usr/bin/python
# -*- coding: utf-8 -*-

from persistent.persistent import Persistent
import pymorphy2
import nltk
from string import punctuation
from random import shuffle, randint
from itertools import islice
import dionysus as dn
import wikipedia
import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
morph = pymorphy2.MorphAnalyzer()


def get_texts_for_lang(lang, n=100):
    wikipedia.set_lang(lang)
    wiki_content = []
    pages = wikipedia.random(n)
    for page_name in pages:
        try:
            page = wikipedia.page(page_name)
        except wikipedia.exceptions.WikipediaException:
            print('Skipping page {}'.format(page_name))
            continue

        wiki_content.append('{}\n{}'.format(page.title, page.content.replace('==', '')))

    return wiki_content


def random_chunk(li, min_chunk=1, max_chunk=3):
    it = iter(li)
    while True:
        nxt = list(islice(it,randint(min_chunk,max_chunk)))
        if nxt:
            yield nxt
        else:
            break


def shuffle_t(text):
    text = flatten(text)
    shuffle(text)
    return list(random_chunk(text, 3, 14))


def scramble(text, path=None):

    if path:
        file = open(path, 'r').read()
        sentences = tokenizer.tokenize(file)

    sentences = tokenizer.tokenize(text)

    split_sent = []

    for s in sentences:
        s = s.translate(str.maketrans('', '', punctuation))
        split_sent.append(s.lower().split())

    for sentence in split_sent:
        for i in range(len(sentence)):
            p = morph.parse(sentence[i])[0]
            sentence[i] = p.normal_form

    shuff_sent = shuffle_t(split_sent)

    a = Persistent(split_sent=split_sent, min_count=5, window=5)
    dgms_n = a.persistent()

    b = Persistent(split_sent=shuff_sent, min_count=5, window=5)
    dgms_sh = b.persistent()

    return dn.wasserstein_distance(dgms_n[0], dgms_sh[0])

if __name__ == '__main__':
    results = []
    texts = get_texts_for_lang('ru')
    for text in texts:
        results.append(scramble(text=text))

    print(np.mean(results))
