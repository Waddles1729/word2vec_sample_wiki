import MeCab
import tensorflow as tf
import numpy as np
import gensim

text = "ひとつ注意ですが、これからネットワークを考えるにあたって実際の単語そのものではなく、ボキャブラリ内でのインデックスとして考えます。"

class Word2vec(object):

    def __init__(self, data):
        self.test_text = data

    def _get_noun_list(self, text):
        tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/')
        # tagger = MeCab.Tagger('mecabrc')
        tagger.parse('')
        node = tagger.parseToNode(text)
        target_parts_of_speech = ('名詞',)
        keywords = []
        while node:
            if node.feature.split(",")[0] in target_parts_of_speech:
                keywords.append(node.surface)
            node = node.next
        return keywords

    def _shaping(self):
        test_text = [word for word in self._get_noun_list(self.test_text)]

        text = [test_text]

        return text

    def _make_corpus(self):
        texts = self._shaping()
        # ディクショナリーの作成
        dictionary = gensim.corpora.Dictionary(texts)
        # corpusの作成
        corpus = [dictionary.doc2bow(text) for text in texts]

        return corpus


w = Word2vec(data=text)
print(w._make_corpus())