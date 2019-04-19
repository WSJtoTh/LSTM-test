#!/usr/bin/env/ python
#_*_ coding:utf-8_*_

#import word2vec
#from gensim.test.utils import common_texts, get_tmpfile
#from gensim.models import Word2Vec
#import sys
import os
from gensim.models import Word2Vec
#from LoadData import loadData
import pickle
from gensim.corpora.dictionary import Dictionary
#reload(sys)
#sys.setdefaultencoding('utf-8')
"""###########################################
状态：词向量的训练效果不尽如人意

#############################################"""

"""
用于从一个文件夹下的多个txt文件中逐行读取训练语句
"""


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
       for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line.split()


def saveWordIndex(model):
    word2vec_dict = Dictionary()
    word2vec_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2v_index = {w: i+1 for i, w in word2vec_dict.items()}#建立词典索引字典，如{'中国':1}，索引从1开始，0目前不存储，以后存储不在字典索引里的词语
    print(w2v_index.keys()[:20])
    w2v_vec = {w: model[w.encode('utf-8')] for w in w2v_index.keys()}#建立词向量字典，如{'中国':['0.01','0.25',......]}
    pickle.dump(w2v_index, open('./w2v_index.pkl', 'w'))
    pickle.dump(w2v_vec, open('./w2v_vec.pkl', 'w'))


def trainWord2Vec():#训练word2vec模型并存储
    """
    trainfile = open('train_words.txt', 'r', encoding='utf-8')
    lines = trainfile.readlines()
    length = len(lines)
    amount = 0
    print("已经完成文件打开")
    count = 0
    sentences = []
    sentence = []
    try:
        for line in lines:
            count = count + 1
            amount = amount + 1
            line = line.split('\n')[0]
            line = line.split('\t')
            for s in line:
                s = s.split(' ')
                sentence.append(s)
            sentences.append(sentence)
            if amount == length:
                print("amount=" + str(amount))
                #My_Sentences.write_into_file(wordFile, faceFile, puncFile, sentFile, count)
                break
            if count >= 100:
                print("count=" + str(count))
                count = count % 100
                #My_Sentences.write_into_file(wordFile, faceFile, puncFile, sentFile, MEMORY_NUM)

    #for line in lines:


    :return:
    """
    sentences = MySentences('D:\pyCharm\LSTM\w2v')
    print(sentences)
    model = Word2Vec(sentences=sentences, size=100, min_count=1, window=5)
    model.save('./word2vec.model')
    vector = model['you']
    v = model.wv['you']
    c = 0
    for key in vector:
        c = c + 1
    print(vector)
    print(str(c))
    print(v)
    model.wv.save_word2vec_format('word_vec.txt', binary=False)
    # model=Word2Vec.load('./word2vec.model')
    #saveWordIndex(model=model)


if __name__ == '__main__':
    #train_file = open('train_words', 'r')
    trainWord2Vec()
    """
    path = get_tmpfile("word2vec.model")
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    model.train(['hello', 'world', 'hello'], total_examples=1, epochs=1)
    vector = model['computer']
    for key in vector:
        print(key)
    
    """


